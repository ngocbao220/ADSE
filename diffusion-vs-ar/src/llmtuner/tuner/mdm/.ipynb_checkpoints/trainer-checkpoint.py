import os
import json
import torch
import numpy as np
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset
from transformers import Trainer
from torch import nn
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.logging import get_logger
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.nn.functional as F

logger = get_logger(__name__)

class IntrospectionHead(torch.nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256): # <--- Lưu ý input_dim=384
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)
                
class CustomDiffusionTrainer(Trainer):
    def __init__(
        self,
        diff_args,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.diff_args = diff_args
        print(self.diff_args)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: Optional[bool] = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        r"""
        Computes diffusion loss.
        """
        final_loss = self.inner_forward(model, inputs)
        return final_loss
    
    def q_sample(self, inputs, t, maskable_mask):
        x_0 = inputs["input_ids"]
        u = torch.rand_like(x_0, dtype=torch.float) # t/T prob to mask
        t_mask = (u < ((t+1) / self.diff_args.diffusion_steps)[:, None]) & maskable_mask
        x_t = x_0.masked_fill(t_mask, self.tokenizer.mask_token_id)
        return x_t, t, t_mask  #  True means it's "MASK" token and should have loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        rewrite prediction_step for eval loss
        """
        model.eval()
        labels = inputs['input_ids'].masked_fill(inputs['src_mask'].bool(), self.tokenizer.pad_token_id)
        with torch.no_grad():
            # import pdb; pdb.set_trace();
            final_loss = self.inner_forward(model, inputs)
            if prediction_loss_only:
                preds = None
            else:
                preds = self.generate_samples(inputs)
        # ignore the source part when calculating metric and saving
        preds = preds.masked_fill(inputs['src_mask'].bool(), self.tokenizer.pad_token_id)
        return final_loss, preds, labels
    
    def save_predictions(
        self,
        predict_results: "PredictionOutput"
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id)
        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for pred, label in zip(decoded_preds, decoded_labels):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))

    def inner_forward(
        self,
        model,
        inputs
    ):
        x = inputs["input_ids"]
        src_mask = inputs["src_mask"].bool()
        batch_size = x.size(0)

        if isinstance(model, DDP):
            vocab_size = model.module.vocab_size
        else:
            vocab_size = model.vocab_size
        num_timesteps = self.diff_args.diffusion_steps

        t = torch.randint(0, num_timesteps, (batch_size, ), device=x.device)
        x_t, t, loss_mask = self.q_sample(inputs, t, maskable_mask=~src_mask)

        attention_mask = torch.ones_like(x_t) 

        logits = model(x_t, t, attention_mask=attention_mask)
        logits = torch.cat([logits[:,0:1], logits[:,:-1]], dim=1)

        loss = F.cross_entropy(logits.reshape(-1, vocab_size), x.reshape(-1), reduction="none").float()   # num_masked samples
        
        loss = loss.masked_fill(~loss_mask.reshape(-1), 0)
        
        if self.diff_args.token_reweighting:
            loss = self.diff_args.alpha * (1 - torch.exp(-loss)) ** self.diff_args.gamma * loss

        if self.diff_args.time_reweighting == 'original':
            weight = 1 / (t+1)[:, None].float()
        elif self.diff_args.time_reweighting == 'linear':
            weight = (num_timesteps - t)[:, None].float()
        else:
            weight = t.new_ones((batch_size, 1)).float()

        weight = weight.expand(loss_mask.size())  # b,n, only [MASK] position have loss
        loss = (loss * weight.reshape(-1)).sum() / loss_mask.sum()   # avg token loss
        return loss

    # def generate_samples(self, inputs):
    #     """
    #     ADAPTIVE INFERENCE (With Verbose Logging): 
    #     Tự động dừng khi đã điền xong, có in log từng bước để theo dõi.
    #     """
    #     self.model.cuda()
    #     self.model.eval()
        
    #     # Biến này để kiểm tra xem có phải đang train hay không. 
    #     # Nếu đang chạy eval/predict thì verbose = True -> In ra log
    #     verbose = not self.is_in_train 
        
    #     # --- CẤU HÌNH ADAPTIVE ---
    #     CONFIDENCE_THRESHOLD = 0.90 
    #     # -------------------------

    #     x = inputs['input_ids'].cuda()
    #     src_mask = inputs['src_mask'].bool().cuda()
    #     attention_mask = torch.ones_like(x) 
    #     batch_size = x.size(0)

    #     maskable_mask = ~src_mask 
    #     steps_taken = 0

    #     # Vòng lặp ngược từ T-1 về 0
    #     for t in range(self.diff_args.diffusion_steps-1, -1, -1):
    #         steps_taken += 1
            
    #         with torch.no_grad():
    #             # 1. Khởi tạo nhiễu (Bước đầu tiên)
    #             if t == self.diff_args.diffusion_steps-1:
    #                 xt = x.masked_fill(maskable_mask, self.tokenizer.mask_token_id)

    #             # --- [LOGGING INPUT] ---
    #             # In ra trạng thái hiện tại của bàn cờ TRƯỚC khi đưa vào model
    #             if verbose:
    #                 # decode danh sách ID thành chuỗi số để dễ nhìn
    #                 print(f"t={t+1}(in):", self.tokenizer.decode(xt.tolist()[0]))

    #             # 2. Forward Model
    #             t_tensor = torch.full((batch_size, ), t, device=x.device)
    #             logits = self.model(xt, t_tensor, attention_mask=attention_mask)
                
    #             # Shift Logits
    #             logits = torch.cat([logits[:,0:1], logits[:,:-1]], dim=1)

    #             # 3. Tính toán
    #             scores = torch.log_softmax(logits, dim=-1)
    #             if hasattr(self.tokenizer, "vocab_size"):
    #                 scores[:,:,self.tokenizer.vocab_size:] = -1000

    #             x0_scores, x0_preds = scores.max(-1)
    #             probs = torch.exp(x0_scores) 

    #             # Cập nhật giá trị dự đoán (x0)
    #             x0 = xt.masked_scatter(maskable_mask, x0_preds[maskable_mask])

    #             # --- [LOGGING OUTPUT] ---
    #             # In ra dự đoán của model (model nghĩ bàn cờ nên trông như thế nào)
    #             if verbose:
    #                 print(f"t={t+1}(out):", self.tokenizer.decode(x0.tolist()[0]))

    #             # 4. --- LOGIC ADAPTIVE ---
    #             high_conf_mask = (probs > CONFIDENCE_THRESHOLD) & maskable_mask
                
    #             # Điền cứng (Unmask) các số tự tin
    #             if high_conf_mask.sum() > 0:
    #                 xt[high_conf_mask] = x0_preds[high_conf_mask]
    #                 maskable_mask[high_conf_mask] = False 
                
    #             # Early Exit
    #             if maskable_mask.sum() == 0:
    #                 if verbose:
    #                     print(f"--> Adaptive Solved at Step {steps_taken}/{self.diff_args.diffusion_steps}. Exiting...")
    #                 break 

    #     return xt

    # def generate_samples(self, inputs):
    #     """
    #     HYBRID ADAPTIVE: Sức mạnh của TopK gốc + Tốc độ của Adaptive
    #     """
    #     self.model.cuda()
    #     self.model.eval()
    #     verbose = not self.is_in_train
        
    #     x = inputs['input_ids'].cuda()
    #     src_mask = inputs['src_mask'].bool().cuda()
    #     attention_mask = torch.ones_like(x) 
    #     batch_size = x.size(0)

    #     # Mask ban đầu (những chỗ cần điền)
    #     init_maskable_mask = maskable_mask = ~src_mask
        
    #     steps_taken = 0

    #     for t in range(self.diff_args.diffusion_steps-1, -1, -1):
    #         steps_taken += 1
            
    #         with torch.no_grad():
    #             # 1. Khởi tạo nhiễu
    #             if t == self.diff_args.diffusion_steps-1:
    #                 xt = x.masked_fill(maskable_mask, self.tokenizer.mask_token_id)
                
    #             # 2. Forward
    #             t_tensor = torch.full((batch_size, ), t, device=x.device)
    #             logits = self.model(xt, t_tensor, attention_mask=attention_mask)
    #             logits = torch.cat([logits[:,0:1], logits[:,:-1]], dim=1) # Logit Shift
                
    #             # 3. Tính Scores
    #             scores = torch.log_softmax(logits, dim=-1)
    #             if hasattr(self.tokenizer, "vocab_size"):
    #                 scores[:,:,self.tokenizer.vocab_size:] = -1000
                
    #             x0_scores, x0_preds = scores.max(-1)
                
    #             # Tạo bản dự đoán x0 tạm thời
    #             x0_preds = xt.masked_scatter(maskable_mask, x0_preds[maskable_mask]) 
                
    #             # 4. --- SỨC MẠNH CỦA BÀI BÁO GỐC (TopK Decoding) ---
    #             # Gọi hàm có sẵn của tác giả để quyết định điền ô nào
    #             if t > 0:
    #                 if self.diff_args.topk_decoding:
    #                     # Hàm này sẽ trả về xt mới với một số ô được unmask theo chiến thuật thông minh
    #                     xt_new = topk_decoding(
    #                         x0_preds, x0_scores, self.diff_args.decoding_strategy,
    #                         maskable_mask, t, self.diff_args.diffusion_steps,
    #                         self.tokenizer.mask_token_id
    #                     )
                        
    #                     # Cập nhật lại maskable_mask (những ô nào chưa điền thì giữ True)
    #                     # Logic: Ô nào vẫn là mask_token thì tức là chưa điền
    #                     current_masks = (xt_new == self.tokenizer.mask_token_id)
    #                     maskable_mask = current_masks & init_maskable_mask
                        
    #                     xt = xt_new
    #                 else:
    #                     # Fallback (Random)
    #                     unmask_prob = 1 / (t+1)
    #                     mask_to_unmask = torch.rand(xt.shape, device=xt.device) < unmask_prob
    #                     mask_to_unmask = mask_to_unmask & maskable_mask
    #                     xt[mask_to_unmask] = x0_preds[mask_to_unmask]
    #                     maskable_mask[mask_to_unmask] = False
    #             else:
    #                 xt = x0_preds # Bước cuối thì lấy luôn kết quả

    #             # 5. --- KẾT HỢP ADAPTIVE (EARLY EXIT) ---
    #             # Kiểm tra: Nếu TopK Decoding đã điền hết sạch ô trống -> Dừng luôn
    #             if maskable_mask.sum() == 0:
    #                 if verbose:
    #                     print(f"--> Hybrid Adaptive Solved at Step {steps_taken}/{self.diff_args.diffusion_steps}")
    #                 break
        
    #     return xt

    def generate_samples(self, inputs):
        """
        MIND-MGDM HYBRID:
        - Sử dụng mô hình MIND đã train để quyết định dừng sớm.
        - Kết hợp Top-K Decoding để đảm bảo độ ổn định.
        """
        self.model.cuda()
        self.model.eval()
        verbose = not self.is_in_train
        
        # --- CẤU HÌNH MIND ---
        MIND_MODEL_PATH = "mind_model.pth" # Đường dẫn file vừa train xong
        MIND_THRESHOLD = 0.85              # Độ tin cậy cần thiết để dừng (0.8 - 0.9)
        # ---------------------
                
        # 1. Load Mạng Nội quan (MIND)
        introspection_net = IntrospectionHead(input_dim=384).cuda()
        try:
            introspection_net.load_state_dict(torch.load(MIND_MODEL_PATH))
            if verbose: print(f"[MIND] Loaded Introspection Network from {MIND_MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Không load được MIND model: {e}. Dùng fallback heuristic.")
            introspection_net = None
        
        if introspection_net: introspection_net.eval()

        # Setup Input
        x = inputs['input_ids'].cuda()
        src_mask = inputs['src_mask'].bool().cuda()
        attention_mask = torch.ones_like(x) 
        batch_size = x.size(0)

        # Mask ban đầu
        init_maskable_mask = maskable_mask = ~src_mask
        steps_taken = 0

        for t in range(self.diff_args.diffusion_steps-1, -1, -1):
            steps_taken += 1
            
            with torch.no_grad():
                # A. Diffusion Forward (Lấy cả Hidden States)
                if t == self.diff_args.diffusion_steps-1:
                    xt = x.masked_fill(maskable_mask, self.tokenizer.mask_token_id)
                
                t_tensor = torch.full((batch_size, ), t, device=x.device)
                
                # --- [QUAN TRỌNG] Lấy Hidden States ---
                # Vì self.model có thể là wrapper, ta truy cập vào backbone bên trong
                # Nếu model bạn là DiffusionModel -> self.model.model là GPT2
                backbone = self.model.model if hasattr(self.model, "model") else self.model
                
                outputs = backbone.transformer(
                    inputs_embeds=backbone.transformer.wte(xt), 
                    return_dict=True, output_hidden_states=True
                )
                hidden_states = outputs.last_hidden_state # [Batch, Seq, 384]
                logits = backbone.lm_head(hidden_states)
                
                # Shift Logits
                logits = torch.cat([logits[:,0:1], logits[:,:-1]], dim=1)
                
                # Tính toán dự đoán cơ bản
                scores = torch.log_softmax(logits, dim=-1)
                if hasattr(self.tokenizer, "vocab_size"):
                    scores[:,:,self.tokenizer.vocab_size:] = -1000
                x0_scores, x0_preds = scores.max(-1)
                x0_full = xt.masked_scatter(maskable_mask, x0_preds[maskable_mask])

                # B. MIND Đánh giá ("Trọng tài")
                can_exit_mask = torch.zeros_like(maskable_mask, dtype=torch.bool)
                
                if introspection_net:
                    mind_probs = introspection_net(hidden_states.float()).squeeze(-1)
                    can_exit_mask = (mind_probs > MIND_THRESHOLD)
                else:
                    # Fallback (nếu không có MIND)
                    probs = torch.exp(x0_scores)
                    can_exit_mask = (probs > 0.9)

                # C. Top-K Schedule ("Người dẫn đường")
                if t > 0 and self.diff_args.topk_decoding:
                    xt_sched = topk_decoding(
                        x0_full, x0_scores, self.diff_args.decoding_strategy,
                        maskable_mask, t, self.diff_args.diffusion_steps,
                        self.tokenizer.mask_token_id
                    )
                    mask_sched = (xt_sched == self.tokenizer.mask_token_id)
                else:
                    mask_sched = torch.zeros_like(maskable_mask, dtype=torch.bool)

                # D. Quyết định cuối cùng (Hybrid Logic)
                # Giữ mask nếu: (TopK bảo giữ) VÀ (MIND bảo chưa tự tin)
                # => Unmask nếu: (TopK bảo nhả) HOẶC (MIND bảo tự tin)
                new_mask = mask_sched & (~can_exit_mask) & init_maskable_mask
                
                # Cập nhật xt
                xt = x0_full.clone()
                xt[new_mask] = self.tokenizer.mask_token_id
                maskable_mask = new_mask

                # E. Early Exit
                if maskable_mask.sum() == 0:
                    if verbose: print(f"--> MIND Solved at Step {steps_taken}/{self.diff_args.diffusion_steps}")
                    break
        
        return xt

    # def generate_samples(self, inputs):
    #     """
    #         select 1/T% tokens to denoise at each step
    #     """
    #     self.model.cuda()
    #     self.model.eval()
    #     verbose = not self.is_in_train
    #     # x = torch.transpose(torch.stack(inputs['input_ids']), 0, 1).cuda()
    #     # src_mask = torch.transpose(torch.stack(inputs['src_mask']), 0, 1).bool().cuda()
    #     x = inputs['input_ids'].cuda()
    #     src_mask = inputs['src_mask'].bool().cuda()
    #     attention_mask = torch.ones_like(x) 
    #     batch_size = x.size(0)

    #     init_maskable_mask = maskable_mask = ~src_mask
        
    #     for t in range(self.diff_args.diffusion_steps-1, -1, -1): # t from T-1 to 0
    #         with torch.no_grad():
    #             if t == self.diff_args.diffusion_steps-1:
    #                 # first forward, all position except src is [M]
    #                 xt = x.masked_fill(maskable_mask, self.tokenizer.mask_token_id)

    #             if verbose:
    #                 print(f"t={t+1}(in):", self.tokenizer.decode(xt.tolist()[0]))

    #             t_tensor = torch.full((batch_size, ), t, device=x.device)
    #             logits = self.model(xt, t_tensor, attention_mask=attention_mask)
    #             logits = torch.cat([logits[:,0:1], logits[:,:-1]], dim=1)

    #             scores = torch.log_softmax(logits, dim=-1)
    #             scores[:,:,self.tokenizer.vocab_size:]=-1000
    #             x0_scores, x0 = scores.max(-1)

    #             #### keep non-[MASK] positions as still
    #             x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
    #             if verbose:
    #                 print(f"t={t+1}(out):", self.tokenizer.decode(x0.tolist()[0]))
                
    #             if t > 0:
    #                 if self.diff_args.topk_decoding:
    #                     xt = topk_decoding(
    #                         x0,
    #                         x0_scores,
    #                         self.diff_args.decoding_strategy,
    #                         init_maskable_mask, 
    #                         t,
    #                         self.diff_args.diffusion_steps,
    #                         self.tokenizer.mask_token_id
    #                     )
    #                 else:
    #                     ## randomly unmask some mask positions as in D3PM
    #                     unmask_prob = 1 / (t+1)
    #                     mask_to_x0 = torch.rand(xt.shape, device=xt.device) < unmask_prob
    #                     # don't unmask somewhere already unmasked
    #                     mask_to_x0 = torch.bitwise_and(mask_to_x0, maskable_mask)
    #                     xt[mask_to_x0] = x0[mask_to_x0]
    #                     maskable_mask.masked_fill_(mask_to_x0, False) 
    #             else:
    #                 xt = x0
    #     return xt

def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len) # + 1e-10
    # cutoff_len = k -> select k + 1 tokens
    masking = _scores < cutoff
    return masking


def topk_decoding(
        x0, 
        x0_scores,
        decoding_strategy,
        init_maskable_mask, 
        t,
        max_step,
        noise
    ):
        # decoding_strategy needs to take the form of "<topk_mode>-<schedule>"
        topk_mode, schedule = decoding_strategy.split("-")

        # select rate% not confident tokens, ~1 -> 0
        if schedule == "linear":
            rate = t / max_step
        elif schedule == "cosine":
            rate = np.cos((max_step-t) / max_step * np.pi * 0.5)
        else:
            raise NotImplementedError
        
        # compute the cutoff length for denoising top-k positions
        cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()
        # set the scores of unmaskable symbols to a large value so that they will never be selected
        _scores_for_topk = x0_scores.masked_fill(~init_maskable_mask, 1000.0)

        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
        elif topk_mode == "deterministic":
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
        else:
            raise NotImplementedError

        ### recovered tokens can also be remasked based on current scores
        masked_to_noise = lowest_k_mask
        if isinstance(noise, torch.Tensor):
            xt = x0.masked_scatter(masked_to_noise, noise[masked_to_noise])
        elif isinstance(noise, (int, float)):
            xt = x0.masked_fill(masked_to_noise, noise)
        else:
            raise NotImplementedError("noise should be either a tensor or a scalar")

        return xt