"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

import random
import math
import torch
import onmt.inputters as inputters
import onmt.utils
from onmt.utils import Statistics
from onmt.utils.distributed import all_gather_list, \
    all_reduce_and_rescale_tensors
from onmt.utils.logging import logger

import onmt.modules, onmt.modules.softmax_extended


def build_trainer(opt, device_id, model, fields,
                  optim, data_type, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    train_loss = onmt.utils.loss.build_loss_compute(
        model, fields["tgt"], opt)
    valid_loss = onmt.utils.loss.build_loss_compute(
        model, fields["tgt"], opt, train=False)

    trunc_size = opt.truncated_decoder
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count
    n_gpu = opt.world_size if device_id >= 0 else 0
    gpu_rank = opt.gpu_ranks[device_id] if device_id >= 0 else 0
    gpu_verbose_level = opt.gpu_verbose_level
    sampling_type = opt.sampling_type
    scheduled_sampling_decay = opt.scheduled_sampling_decay
    scheduled_sampling_k = opt.scheduled_sampling_k
    scheduled_sampling_c = opt.scheduled_sampling_c
    scheduled_sampling_limit = opt.scheduled_sampling_limit
    mixture_type = opt.mixture_type
    topk_value = opt.topk_value
    peeling_back = opt.peeling_back
    twopass = opt.decoder_type == 'transformer'
    passone_nograd = (not opt.transformer_passone) or \
                    (opt.transformer_passone and opt.transformer_passone == 'nograd')
    scheduled_activation = opt.transformer_scheduled_activation
    scheduled_softmax_alpha = opt.transformer_scheduled_alpha

    report_manager = onmt.utils.build_report_manager(opt)
    trainer = onmt.Trainer(model, train_loss, valid_loss, optim, trunc_size,
                           shard_size, data_type, norm_method,
                           grad_accum_count, n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           model_saver=model_saver,
                           sampling_type=sampling_type,
                           scheduled_sampling_decay=scheduled_sampling_decay,
                           scheduled_sampling_k=scheduled_sampling_k,
                           scheduled_sampling_c=scheduled_sampling_c,
                           scheduled_sampling_limit=scheduled_sampling_limit,
                           mixture_type=mixture_type,
                           topk_value=topk_value,
                           peeling_back=peeling_back,
                           twopass=twopass,
                           passone_nograd=passone_nograd,
                           scheduled_activation=scheduled_activation,
                           scheduled_softmax_alpha=scheduled_softmax_alpha)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 gpu_verbose_level=0, report_manager=None, model_saver=None,
                 sampling_type="teacher_forcing",
                 scheduled_sampling_decay="exp",
                 scheduled_sampling_k=1.0, scheduled_sampling_c=1.0,
                 scheduled_sampling_limit=0.0,
                 mixture_type='none',
                 topk_value=1,
                 peeling_back='none',
                 twopass=False,
                 passone_nograd='nograd',
                 scheduled_activation='softmax',
                 scheduled_softmax_alpha='1.0'):
        self.model = model
        self._train_loss = train_loss
        self._valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self._norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver
        self._sampling_type = sampling_type
        self._scheduled_sampling_decay = scheduled_sampling_decay
        self._scheduled_sampling_k = scheduled_sampling_k
        self._scheduled_sampling_c = scheduled_sampling_c
        self._scheduled_sampling_limit = scheduled_sampling_limit
        self._mixture_type = mixture_type
        self._k = topk_value
        self._peeling_back = peeling_back
        self._twopass = twopass
        self._passone_nograd = passone_nograd
        if scheduled_activation == "sparsemax":
            self._scheduled_activation_function = onmt.modules.sparse_activations.Sparsemax(dim=-1)
        elif scheduled_activation == "gumbel":
            self._scheduled_activation_function = onmt.modules.softmax_extended.GumbelSoftmax(dim=-1, alpha=scheduled_softmax_alpha)
        elif scheduled_activation == "softmax_temp":
            self._scheduled_activation_function = onmt.modules.softmax_extended.SoftmaxWithTemperature(dim=-1, alpha=scheduled_softmax_alpha)
        else:
            self._scheduled_activation_function = torch.nn.Softmax(dim=-1)

        assert grad_accum_count == 1  # disable grad accumulation

        self.model.train()

    def train(self, train_iter, valid_iter, train_steps, valid_steps):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):
        """
        logger.info('Start training...')

        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:

            # there should be only one loop
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):
                    if self.gpu_verbose_level > 1:
                        logger.info("GpuRank %d: index: %d accum: %d"
                                    % (self.gpu_rank, i, accum))

                    true_batchs.append(batch)

                    if self._norm_method == "tokens":
                        num_tokens = batch.tgt[1:].ne(
                            self.train_loss.padding_idx).sum()
                        normalization += num_tokens.item()
                    else:
                        normalization += batch.batch_size
                    accum += 1
                    if accum == self.grad_accum_count:

                        batch_teacher_forcing_ratio = \
                            self._calc_teacher_forcing_ratio(step)

                        # print('TRANSF_GRAD: step: ', step)

                        self._train_batch(
                            batch, normalization, total_stats, report_stats,
                            batch_teacher_forcing_ratio, step
                        )

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim.learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if (step % valid_steps == 0):
                            if self.gpu_verbose_level > 0:
                                logger.info('GpuRank %d: validate step %d'
                                            % (self.gpu_rank, step))
                            valid_stats = self.validate(valid_iter)
                            if self.gpu_verbose_level > 0:
                                logger.info('GpuRank %d: gather valid stat \
                                            step %d' % (self.gpu_rank, step))
                            valid_stats = self._maybe_gather_stats(valid_stats)
                            if self.gpu_verbose_level > 0:
                                logger.info('GpuRank %d: report stat step %d'
                                            % (self.gpu_rank, step))
                            self._report_step(self.optim.learning_rate,
                                              step, valid_stats=valid_stats)

                        if self.gpu_rank == 0:
                            self._maybe_save(step)
                        step += 1
                        if step > train_steps:
                            break
            if self.gpu_verbose_level > 0:
                logger.info('GpuRank %d: we completed an epoch \
                            at step %d' % (self.gpu_rank, step))

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        with torch.no_grad():
            stats = onmt.utils.Statistics()

            for batch in valid_iter:
                src = inputters.make_features(batch, 'src', self.data_type)
                if self.data_type == 'text':
                    _, src_lengths = batch.src
                elif self.data_type == 'audio':
                    src_lengths = batch.src_lengths
                else:
                    src_lengths = None

                tgt = inputters.make_features(batch, 'tgt')

                # F-prop through the model.
                outputs, attns = self.model(src, tgt, src_lengths)
 
                # Compute loss.
                batch_stats = self._valid_loss.monolithic_compute_loss(
                    batch, outputs, attns)

                # Update statistics.
                stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def _calc_teacher_forcing_ratio(self, step):
        if self._sampling_type == "teacher_forcing":
            return 1.0
        elif self._sampling_type == "scheduled":  # scheduled sampling
            if self._scheduled_sampling_decay == "exp":
                scheduled_ratio = self._scheduled_sampling_k ** step
            elif self._scheduled_sampling_decay == "sigmoid":
                if step / self._scheduled_sampling_k > 700:
                    scheduled_ratio = 0
                else:
                    scheduled_ratio = self._scheduled_sampling_k / (
                            self._scheduled_sampling_k
                            + math.exp(step / self._scheduled_sampling_k)
                            )
            else:  # linear 
                scheduled_ratio = self._scheduled_sampling_k - \
                                    self._scheduled_sampling_c * step
            scheduled_ratio = max(self._scheduled_sampling_limit, scheduled_ratio)
            return scheduled_ratio
        else:  # always sample from the model predictions
            return 0.0

    def _train_batch(self, batch, normalization, total_stats, report_stats,
                     teacher_forcing_ratio,step=None):
        target_size = batch.tgt.size(0)
        trunc_size = self.trunc_size if self.trunc_size else target_size

        src = inputters.make_features(batch, 'src', self.data_type)
        if self.data_type == 'text':
            _, src_lengths = batch.src
            report_stats.n_src_words += src_lengths.sum().item()
        elif self.data_type == 'audio':
            src_lengths = batch.src_lengths
        else:
            src_lengths = None

        tgt_outer = inputters.make_features(batch, 'tgt')

        dec_state = None
        emb_weights = None
        top_k_tgt = None
        tf_gate_value = None

        for j in range(0, target_size-1, trunc_size):
            # 1. Create truncated target.
            tgt = tgt_outer[j: j + trunc_size]

            # 2. F-prop all but generator.
            self.model.zero_grad()
            if teacher_forcing_ratio >= 1:
                outputs, attns = self.model(src, tgt, src_lengths)
                # bpop important note: the output layer has not been applied to
                # these outputs yet, so you can't use the outputs tensor by
                # itself to get the model's next prediction.
            elif self._twopass:
                tf_tgt_section = round(target_size*teacher_forcing_ratio)
                if tf_tgt_section >= target_size:
                    # The standard model
                    outputs, attns = self.model(src, tgt, src_lengths)
                else:
                    tgt = tgt[:-1]

                    # 1. Go through the encoder
                    enc_state, memory_bank, lengths = \
                            self.model.encoder(src, src_lengths)
                    self.model.decoder.init_state(src, memory_bank, enc_state)

                    # This part can be with grad or no_grad
                    if self._passone_nograd:
                        with torch.no_grad():
                            outputs, attns = self.model.decoder(tgt, memory_bank,
                                            memory_lengths=lengths)
                            logits = self.model.generator[0](outputs)

                    else:
                        outputs, attns = self.model.decoder(tgt, memory_bank,
                                            memory_lengths=lengths)                        

                        logits = self.model.generator[0](outputs)

                    # 2. Get the embeddings from the model predictions
                    if self._mixture_type and 'topk' in self._mixture_type:
                        k = self._k
                        emb_weights, top_k_tgt = logits.topk(k, dim=-1)

                        # Needed for getting the embeddings
                        top_k_tgt = top_k_tgt.unsqueeze(-2)

                        # k_embs: batch x k x emb size
                        k_embs = self.model.decoder.embeddings(top_k_tgt,step=0).transpose(2,3) 
                        # weights: batch x sequence length x k x 1
                        # Normalize the weights
                        emb_weights /= emb_weights.sum(dim=-1).unsqueeze(2)
                        weights = emb_weights.unsqueeze(3) 
                        emb_size = k_embs.shape[2]
                        embeddings = self.model.decoder.embeddings(top_k_tgt,step=0)
                        model_prediction_emb = torch.bmm(k_embs.view(-1, emb_size, k), weights.view(-1, k, 1)) #.transpose(0, 1)
                        model_prediction_emb = model_prediction_emb.view(batch.batch_size, -1, emb_size).transpose(0,1)
                    elif self._mixture_type and 'all' in self._mixture_type:
                        logits = self._scheduled_activation_function(logits)

                        # weights = logits
                        # Get the indices of all words in the vocabulary
                        ind = torch.cuda.LongTensor([i for i in range(logits.shape[2])])
                        # We need this format of the indices to ge tht embeddings from the decoder
                        ind = ind.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                        embeddings = self.model.decoder.embeddings(ind,step=0)[0][0]

                        # The predicted embedding is the weighted sum of the words in the vocabulary
                        model_prediction_emb = torch.matmul(logits, embeddings)
                    else:
                        # Just get the argmax from the model predictions
                        logits = self.model.generator[1](logits)
                        model_predictions = logits.argmax(dim=2).unsqueeze(2)
                        model_prediction_emb = self.model.decoder.embeddings(model_predictions)

                    # Get the embeddings of the gold target sequence.
                    tgt_emb = self.model.decoder.embeddings(tgt)

                    # 3. Combine the gold target with the model predictions
                    if self._peeling_back == 'strict':
                        # Combine the two sequences with peelingback
                        # First part from the gold, second part from the model predictions
                        tf_tgt_emb = torch.cat((tgt_emb[:tf_tgt_section], \
                                model_prediction_emb[tf_tgt_section:]))
                    else:
                        # Use scheduled sampling - on each step decide 
                        # whether to use teacher forcing or model predictions.
                        tf_tgt_emb = [tgt_emb[i].unsqueeze(0) \
                                if random.random() <= teacher_forcing_ratio else \
                                model_prediction_emb[i].unsqueeze(0) for i in range(target_size-1)]
                        # tf_tgt_emb.append(tgt_emb[-1].unsqueeze(0))
                        tf_tgt_emb = torch.cat((tf_tgt_emb), dim=0)
                    # Rerun the forward pass with the new target context
                    outputs, attns = self.model.decoder(tgt, memory_bank,
                                    memory_lengths=lengths, step=None, tf_emb=tf_tgt_emb)

            else:
                # RNN
                enc_state, memory_bank, lengths = \
                    self.model.encoder(src, src_lengths)
                self.model.decoder.init_state(src, memory_bank, enc_state)

                tgt_input = tgt[0].unsqueeze(0)
                out_list = []
                for i in range(1, len(tgt)):
                    dec_out, attns = self.model.decoder(
                        tgt_input, memory_bank,
                        memory_lengths=lengths,
                        top_k_tgt=top_k_tgt,
                        emb_weights=emb_weights,
                        mixture_type=self._mixture_type,
                        tf_gate_value=tf_gate_value,
                        tf_ratio=teacher_forcing_ratio)
                    out_list.append(dec_out)

                    # flip a coin for teacher forcing
                    use_tf = random.random() < teacher_forcing_ratio

                    # define whether the teacher forcing
                    # depends on the position
                    # in the sequence
                    if self._peeling_back == 'strict':
                        # strick peeling back means that the given teacher
                        # forcing ratio
                        # defines the part of the sequence that uses
                        # teacher forcing
                        use_tf = float(j)/target_size < teacher_forcing_ratio

                    if use_tf:
                        tgt_input = tgt[i].unsqueeze(0)
                        emb_weights = None
                        top_k_tgt = None
                    elif self._mixture_type is None:
                        # plain-old use of the model's own prediction
                        logits = self.model.generator[0](dec_out)
                        tgt_input = logits.argmax(dim=2).unsqueeze(2)
                    else:
                        tgt_input = tgt[i].unsqueeze(0)
                        # mixture of target embeddings based on output probs
                        gen_out = torch.exp(self.model.generator(dec_out))
                        if 'topk' in self._mixture_type:
                            k = self._k
                        else:
                            # getting the k for the sparsemax case,
                            # where k is chosen accordding to the number of
                            # nonzero elements.
                            k = (gen_out > 0).sum(-1).max().item()
                        emb_weights, top_k_tgt = gen_out.topk(k, dim=-1)
                        if 'topk' in self._mixture_type:
                            # normalize the weights (not necessary for
                            # the all case because the k will include all the
                            # probability mass)
                            # (might be better to do something with softmax,
                            # bpop is not sure)
                            emb_weights /= emb_weights.sum(dim=-1).unsqueeze(2)

                        if 'tf_gate' in self._mixture_type:
                            tf_gate_value = self.model.tf_gate(gen_out)

                outputs = torch.cat(out_list)

            # 3. Compute loss in shards for memory efficiency.
            # print('memory sizes:', trunc_size, self.shard_size, len(batch), outputs.shape)
            # print('before loss:', outputs.shape)
            # print('batch:', batch)
            batch_stats = self._train_loss.sharded_compute_loss(
                batch, outputs, attns, j,
                trunc_size, self.shard_size, normalization)
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            # Multi GPU gradient gather
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad and p.grad is not None]
                all_reduce_and_rescale_tensors(grads, 1.0)
            self.optim.step()

            # If truncated, don't backprop fully.
            if dec_state is not None:
                dec_state.detach()

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)

    def _norm(self, batch):
        if self._norm_method == "tokens":
            norm = batch.tgt[1:].ne(self._train_loss.padding_idx).sum()
        else:
            norm = batch.batch_size
        if self.n_gpu > 1:
            norm = sum(all_gather_list(norm))
        return norm
