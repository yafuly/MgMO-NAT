from email.policy import default
from json import decoder
import math
from multiprocessing.sharedctypes import SynchronizedString
import torch
from re import L
from dataclasses import dataclass


import torch.nn.functional as F
from torch import Tensor
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from .nat_loss import LabelSmoothedDualImitationCriterion
from fairseq.data.dictionary import BOS, EOS, PAD

from omegaconf import II
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from .modified_bleu import sentence_gleu_mod
from .chrf import computeSentChrF
from rouge import Rouge
from sacrebleu.metrics import TER

@register_criterion("nat_mgmo_loss")
class NatMGMOTrainingCriterion(LabelSmoothedDualImitationCriterion):
    def __init__(self, task, label_smoothing, n_gram=6, method='gleu', n_sample=10, temperature=1):
        super().__init__(task, label_smoothing)
        self.args = task.args
        self.tgt_dict = task.tgt_dict
        self.tokenizer = task.tokenizer
        self.n_gram = self.args.n_gram
        self.reward_method = self.args.reward_method
        self.n_sample = self.args.n_sample
        self.alpha = self.args.reward_alpha
        self.reward_factor = self.args.reward_factor
        self.len_loss = self.args.len_loss
        self.with_forward = self.args.with_forward 
        self.temperature = temperature
        self.use_original_mask = self.args.use_original_mask
        self.greedy = self.args.greedy
        self.null_input = self.args.null_input
        self.rm_scale = self.args.rm_scale
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )

        parser.add_argument(
            "--n-gram",
            default=6,
            type=int,
            metavar="I",
            help="n-gram for gleu or bleu score",
        )
        parser.add_argument(
            "--reward-method",
            default="gleu",
            type=str,
            help="method, e.g., gleu or bleu score"
        )
        parser.add_argument(
            "--n-sample",
            default=1,
            type=int,
            help="number of samples for Q distribution"
        )

        parser.add_argument(
            "--reward-alpha",
            default=1,
            type=float,
            help="alpha"
        )
        parser.add_argument(
            "--reward-factor",
            default=1,
            type=float,
            help="alpha"
        )

        parser.add_argument(
            "--token-greedy",
            default=False,
            action="store_true",
            help="alpha"
        )      

        parser.add_argument(
            "--len-loss",
            default=False,
            action="store_true",
            help="alpha"
        )      

        parser.add_argument(
            '--with-forward',
            action="store_true",
            default=False
        )

        parser.add_argument(
            '--temperature',
            type=float,
            default=1,
        )

        parser.add_argument(
            '--debpe-before',
            action="store_true",
            default=False
        )

        parser.add_argument(
            '--use-original-mask', # PF mode
            action="store_true",
            default=False
        )

        parser.add_argument(
            '--greedy',
            action="store_true",
            default=False
        )        

        parser.add_argument(
            '--null-input', # NP mode
            action="store_true",
            default=False
        )     

        parser.add_argument(
            '--rm-scale',
            type=float,
            default=1.,
        )        

    def get_risk(self, y_indices, ref_indices, mask=None, pad_mask=None):
        # This method gets the reward based on the sampling result and reference sentence.
        # For now, we uses GLEU in NLTK, but you can used your own well-defined reward function.
        # In addition, GLEU is variation of BLEU, and it is more fit to reinforcement learning.
        sf = SmoothingFunction()
        score_func = {
            'gleu':  lambda ref, hyp: sentence_gleu([ref], hyp, max_len=self.n_gram),
            'bleu1': lambda ref, hyp: sentence_bleu([ref], hyp,
                                                    weights=[1./self.n_gram] * self.n_gram,
                                                    smoothing_function=sf.method1),
            'bleu2': lambda ref, hyp: sentence_bleu([ref], hyp,
                                                    weights=[1./self.n_gram] * self.n_gram,
                                                    smoothing_function=sf.method2),
            'bleu4': lambda ref, hyp: sentence_bleu([ref], hyp,
                                                    weights=[1./self.n_gram] * self.n_gram,
                                                    smoothing_function=sf.method4),
            'meteor': lambda ref, hyp: meteor_score([ref], hyp),
            "gleu_mod": lambda ref, hyp: sentence_gleu_mod([ref], hyp, max_len=self.n_gram),
            "rouge2": lambda ref, hyp: Rouge().get_scores(" ".join(ref), " ".join(hyp))[0]["rouge-2"]["f"],
            "rougel": lambda ref, hyp: Rouge().get_scores(" ".join(ref), " ".join(hyp))[0]["rouge-l"]["f"],
            "chrf": lambda ref, hyp: computeSentChrF([ref], hyp),
            "ter": lambda ref, hyp: - TER().corpus_score([hyp],[[ref]] ).score / 100,
        }[self.reward_method]


        # |y_indices| = (B, T1)
        # |ref_indices| = (B, T2) 
        def decode(toks, escape_unk=False, specify_unk="", debpe="@@ "):
            if not specify_unk:
                unk_string = ("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP")
            else:
                unk_string = specify_unk
            s = self.tgt_dict.string(
                toks.int().cpu(),
                debpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=unk_string,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s


        def _rm_beos(l):
            bos = str(BOS)
            eos = str(EOS)
            if bos in l:
                l.remove(bos)
            if eos in l:
                l.remove(eos)


        with torch.no_grad():
            risks = []

            for b in range(ref_indices.size(0)):
                ref, hyp = [], []
                sm = pad_mask[b]
                for t in range(ref_indices.size(-1)):
                    if t<mask.size(-1) and not mask[b][t]: # only focus on predicted tokens on masked inputs
                        i = 3
                    else:
                        i = ref_indices[b][t]
                    ref += [str(int(i))]
                    if i == EOS:
                        break
                for t in range(y_indices.size(-1)):
                    tm = sm[t]
                    if not tm: # stop by mask
                        break
                    if not mask[b][t]:
                        i = 3
                    else:
                        i = y_indices[b][t]
                    hyp += [str(int(i))]
                    if i == EOS:
                        break

                # remove start & end token
                _rm_beos(ref)
                _rm_beos(hyp)
                if self.reward_method == 'chrf' or self.reward_method == 'ter': # need character for chrf
                    ref = torch.tensor([int(e) for e in ref])
                    hyp = torch.tensor([int(e) for e in hyp])
                    ref = decode(ref, specify_unk="u")
                    hyp = decode(hyp, specify_unk="u")
                
                risks += [- score_func(ref, hyp) * 100.]

            risks = torch.FloatTensor(risks).to(ref_indices.device)

            return risks



    def get_loss(self):
        return

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        if 'glat' in sample:
            glat = sample['glat']
        else:
            glat = None

        # taking sampling process
        outputs = model.sample(src_tokens, src_lengths, prev_output_tokens, tgt_tokens, self.n_sample, with_forward=self.with_forward, temperature=self.temperature, glat=glat, null_input=self.null_input, rm_scale=self.rm_scale)
        y_indice, log_token_prob = model.sample_logits(outputs['word_ins']['out'], greedy=self.greedy)


        prev_output_tokens_mask = outputs['word_ins']['mask']
        if 'pad_mask' in outputs['word_ins']:
            pad_mask = outputs['word_ins']['pad_mask'] # partial prediction for cmlm
            if self.use_original_mask:
                prev_output_tokens_mask = pad_mask # full prediction for cmlm
        else:
            pad_mask = prev_output_tokens_mask


        # calculate risk
        risks = self.get_risk(y_indice, tgt_tokens.repeat_interleave(self.n_sample,0), mask=prev_output_tokens_mask, pad_mask=pad_mask)
        risks = risks.view(-1,self.n_sample)


        # renormalize by Q-ditribution (default)
        log_length_prob = outputs['length']['log_length_prob']

        log_token_prob = log_token_prob.squeeze(-1).masked_fill_(~prev_output_tokens_mask, 0)
        # derive sentence prob
        log_sent_prob = torch.sum(log_token_prob, -1) 
        # add log length prob
        log_sent_prob = log_sent_prob.view(-1,self.n_sample) # reshape to batch_size x num_sample
        log_sent_prob += log_length_prob
        log_sent_prob *= self.alpha


        # log_prob to probability
        sent_prob = (log_sent_prob).exp()
        # normalize in sample space
        sent_prob = sent_prob / torch.sum(sent_prob, -1, keepdim=True)
        avg_risk = torch.sum(risks * sent_prob, dim=-1)
        batch_loss = torch.sum(avg_risk)
        

        # |y_out| = (B, T, H)
        # |y_indice| = (B, T)
        # |tgt_tokens| = (B, T)
        # |actor_reward| = (B)
        outputs['word_ins']['loss'] = batch_loss
        outputs['word_ins']['factor'] = self.reward_factor
        outputs['word_ins']['nll_loss'] = False
        # ignore length loss temporarily
        if not self.len_loss:
            del outputs['length']

        # incorporate forward loss if needed
        if self.with_forward:
            forward_loss = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens,)  
            outputs["fw_loss"] = forward_loss["word_ins"]



        losses, nll_loss = [], []

        for obj in outputs:
            if obj.startswith('glat'):
                continue
            if outputs[obj].get("loss", None) is None:
                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                    normalize=outputs[obj].get("normalize", False),
                )
            else:

                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
   
            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 else loss.new_tensor(0)
        
        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "avg_risk": avg_risk.mean(),
        }

        # Other tensorboard items go here
        if "stat:softcopy_temp" in outputs:
            logging_output["softcopy_temp"] = outputs["stat:softcopy_temp"]

        # if "stat:latent_dist" in outputs:
        #     logging_output["latent_dist"] = outputs["stat:latent_dist"]["latent_dist"]

        for l in losses:
            
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss*factor, "factor": factor}

    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0, normalize=False
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """
        


        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]


        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1) if not normalize else outputs

            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")


            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                    nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))
        avg_risk = utils.item(sum(log.get("avg_risk", 0) for log in logging_outputs))
        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )
        metrics.log_scalar(
            "avg_risk", avg_risk / sample_size , round=3
        )

        # other tensorboard items calculated here
        if "softcopy_temp" in logging_outputs[0]:
            softcopy_temp = utils.item(sum([log.get("softcopy_temp", 0) for log in logging_outputs])/len(logging_outputs))
            metrics.log_scalar(
                "softcopy_temp",
                softcopy_temp,
                sample_size, round=4
            )
        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )