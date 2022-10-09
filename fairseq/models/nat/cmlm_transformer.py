# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements:
Ghazvininejad, Marjan, et al.
"Constant-time machine translation with conditional masked language models."
arXiv preprint arXiv:1904.09324 (2019).
"""
import torch
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat.nat import NATransformerModel
from fairseq.utils import new_arange


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


@register_model("cmlm_transformer")
class CMLMNATransformerModel(NATransformerModel):
    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, train_ratio=None, **kwargs
    ):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )

        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )
        word_ins_mask = prev_output_tokens.eq(self.unk)

        return {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": word_ins_mask,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor,
            },
        }

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):

        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.eq(self.unk)
        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
        ).max(-1)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        if history is not None:
            history.append(output_tokens.clone())

        # skeptical decoding (depend on the maximum decoding steps.)
        if (step + 1) < max_step:
            skeptical_mask = _skeptical_unmasking(
                output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
            )

            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

    def sample_logits( 
        self, logits, greedy=False, **kwargs
    ):
        # |logits| = (B, T, V) = log_softmax(hidden_outs)
        B, T, V = logits.size()
        prob = logits.exp()
        if not greedy:
            sample = torch.multinomial(prob.view(B*T,V), 1).view(B, 
            T)

        else:
            sample = prob.view(B*T,V).argmax(dim=-1).view(B,T)
        log_token_prob = torch.gather(prob, 2, sample[...,None]).log()
        return sample, log_token_prob

            
    def sample(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, n_sample=1, normalize=True, train_ratio=None, with_forward=False, temperature=1, glat=None, greedy=False, null_input=False, rm_scale=1.0, **kwargs
    ):
        outputs = {}

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        length_out = self.decoder.forward_length(
            normalize=normalize, encoder_out=encoder_out
        )

        def _random_mask(target_tokens):
            pad = self.pad
            bos = self.bos
            eos = self.eos
            unk = self.unk

            target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
            )
            target_score = target_tokens.clone().float().uniform_() 
            target_score.masked_fill_(~target_masks, 2e6)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_() * rm_scale
            target_length = target_length + 1  # make sure to mask at least one token
            
            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk
            )
            return prev_target_tokens



        ## Sampling
        # length sampling 
        sampled_length_tgt, log_length_prob, golden_length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, num_samples=n_sample, tgt_tokens=tgt_tokens)

        # reconstruct prev_output_tokens based on sampled lengths
        # |sampled_length_tgt| = (B, S)
        B, S = sampled_length_tgt.size()
        # tgt_tokens: |B| x |T|
        # prev_tgt_tokens: |BS| x |T|
        prev_output_tokens = tgt_tokens.repeat_interleave(S,0)
        prev_output_tokens = _random_mask(prev_output_tokens)


        # force to sampled lengths
        sampled_length_tgt = sampled_length_tgt.view(B*S)
        max_leng = sampled_length_tgt.max()
        len_diff = max_leng - prev_output_tokens.size(1)
        if len_diff > 0:
            len_aug = torch.full((B*S,len_diff),self.unk).cuda()
            prev_output_tokens = torch.cat((prev_output_tokens,len_aug),dim=1)
        else:
            prev_output_tokens = prev_output_tokens[:,:max_leng]

        ins_idx = torch.arange(B*S).cuda()
        eos_idx = torch.stack((ins_idx, sampled_length_tgt-1)).t()
        
        # reassign eos and pad values according to sampled length
        _tmp = torch.zeros([B*S,max_leng]).long().cuda()
        ins_idx = torch.arange(B*S).cuda()
        eos_idx = torch.stack((ins_idx, sampled_length_tgt-1)).t()
        _tmp[eos_idx[:,0],eos_idx[:,1]] = self.eos
        idx = torch.cumsum((_tmp==2),1)
        sent_idx = idx < 1
        pad_idx = ~sent_idx


        prev_output_tokens[pad_idx] = self.pad
        prev_output_tokens[eos_idx[:,0],eos_idx[:,1]] = self.eos
        word_ins_mask = prev_output_tokens.eq(self.unk)
        pad_mask = prev_output_tokens.ne(self.pad)

        # set to null input in needed (NP case)
        if null_input:
            prev_output_tokens = prev_output_tokens.masked_fill(pad_mask, self.unk)# fill non-pad position with unk
            prev_output_tokens[eos_idx[:,0],eos_idx[:,1]] = self.eos

        # duplicate encoder_out by n_sample times
        encoder_out['encoder_out'] = [encoder_out['encoder_out'][0].repeat_interleave(n_sample,1)]
        encoder_out['encoder_padding_mask'] = [encoder_out['encoder_padding_mask'][0].repeat_interleave(n_sample,0)]


        
        # decoding
        word_ins_out = self.decoder(
            normalize=normalize,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            temperature=temperature,
        )
        

        outputs["word_ins"] = {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": word_ins_mask,
                "pad_mask": pad_mask,
                "ls": self.args.label_smoothing,    
                "nll_loss": True,
                "normalize": normalize,
            }
        outputs["length"] = {
                "out": length_out,
                "tgt": golden_length_tgt,
                "factor": self.decoder.length_loss_factor,
                "log_length_prob": log_length_prob,
                "normalize": normalize,
            }
        # outputs =  {
        #     "word_ins": {
        #         "out": word_ins_out,
        #         "tgt": tgt_tokens,
        #         "mask": prev_output_tokens.ne(self.pad),
        #         "ls": self.args.label_smoothing,    
        #         "nll_loss": True,
        #         "normalize": normalize,
        #     },
        #     "length": {
        #         "out": length_out,
        #         "tgt": golden_length_tgt,
        #         "factor": self.decoder.length_loss_factor,
        #         "log_length_prob": log_length_prob,
        #         "normalize": normalize,
        #     },
        # }
        return outputs


@register_model_architecture("cmlm_transformer", "cmlm_transformer")
def cmlm_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.ngram_predictor = getattr(args, "ngram_predictor", 1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture("cmlm_transformer", "cmlm_transformer_wmt_en_de")
def cmlm_wmt_en_de(args):
    cmlm_base_architecture(args)
