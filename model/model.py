import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from transformers import LogitsProcessorList, StoppingCriteriaList
from typing import Callable, Optional, Union
import types
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
import warnings
from transformers.generation.utils import GENERATION_MODES_MAPPING, GenerationMixin
from tqdm import tqdm
from PIL import Image
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaCausalLMOutputWithPast,
)
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig

from huggingface_hub import hf_hub_download, login
from transformers.utils import logging
login(token="")

logger = logging.get_logger(__name__)
# Hard requirement for the linear layers
# class_counts = [1243, 653, 1788, 1475] 
# total_samples = sum(class_counts) # 5159
# num_classes = len(class_counts)   # 4

# # 2. Calculate weights (Inverse Class Frequency)
# # Formula: Total / (Num_Classes * Class_Count)
# class_weights = [total_samples / (num_classes * c) for c in class_counts]
# weight_tensor = torch.tensor(class_weights).float().to(device)
# criterion = nn.CrossEntropyLoss(weight=weight_tensor)



def focal_loss(logits, labels, gamma=2.0, alpha=0.25):
    ce_loss = F.cross_entropy(logits, labels, reduction="none")
    pt = torch.exp(-ce_loss)
    return (alpha * (1 - pt) ** gamma * ce_loss).mean()



class MultiTaskingPaligemmaModel(nn.Module):
    def __init__(self, class_counts=torch.tensor([1,2], dtype=torch.float), num_classes=4, model_name= "google/paligemma2-10b-pt-224",
                 max_seq_len = 2048, token_k=50, training_mode=False,
                 focal_gamma=2.0, criterion_type = "weighted_ce", device_map="auto", strategy="freeze", lm_use=True):
        super().__init__()
        '''
        if lm_use:True
        anissssshhhhhh/paligemma_effective_layer_generation__finetune_toxic_tags
        '''
        
        self.model_name = model_name
        self.training_mode = training_mode
        self.processor = PaliGemmaProcessor.from_pretrained(
            self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.focal_gamma = focal_gamma
        self.lm_use = lm_use
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.k = token_k
        self.criterion_type = criterion_type
        lora_config = LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
        self.strategy = strategy
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )
        self.base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name, quantization_config=bnb_config, device_map=device_map
        )
        for mode_method in GENERATION_MODES_MAPPING.values():
            method = getattr(self.base_model, mode_method, None)
            if method is not None:
                setattr(self, mode_method, types.MethodType(method.__func__, self))
        # self.final_norm = self.base_model.model.language_model.norm   
        # class_counts = torch.tensor([1243, 653, 1788, 1475], dtype=torch.float)
        total_samples = class_counts.sum()
        class_weights = total_samples / (len(class_counts) * class_counts)

        self.register_buffer("class_weights", class_weights)
        self.ce_criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.base_model = get_peft_model(self.base_model, lora_config)
        if self.strategy == "freeze":
            for param in self.base_model.parameters():
                param.requires_grad = False

            
        self.hidden_size = self.base_model.config.text_config.hidden_size
    
        # YOUR ARCHITECTURE: (2048 -> 1024 -> 1)
        self.seq_compression = nn.Sequential(
            nn.Linear(self.max_seq_len, 1024),
            nn.GELU(),
            nn.Linear(1024, 1)
        )
        
        # self.k_compression = nn.Sequential(
        #     nn.Linear(self.k, 100),
        #     nn.GELU(),
        #     nn.Linear(100, 1)
        # )
        self.label_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.LayerNorm(self.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size // 4, num_classes),
        )        
    
    def collate_fn(self, examples):
        
        texts = [ self.processor.tokenizer.additional_special_tokens[0] + str(example[1])for example in examples]
        labels = [str(example[2]) for example in examples]
        images = [example[0] for example in examples]
        class_labels = [example[3] for example in examples]

        tokens = self.processor(
            text=texts,
            images=images,
            suffix=labels,
            return_tensors="pt",
            padding="max_length",  
            truncation=True,
            max_length=self.MAX_SEQ_LEN
        )

        inputs = {k: v for k, v in tokens.items()}
        inputs["class_labels"] = torch.tensor(class_labels, dtype=torch.long)
        return inputs
    
    def confident_embedding_per_token_extraction(self, outputs):
        with torch.no_grad():
            layer_entropies = []
            
            for layer_h in outputs.hidden_states:
                
                # last layer -> norm 
                normed = self.base_model.model.language_model.norm(layer_h) # [Batch, Seq, Hidden]
                
                # 1. Cast to Float32 to prevent NaNs in Softmax
                logits = self.base_model.lm_head(normed).float()  # batch, seq, hiddensize, [hiddensize, vocab]  
                probs = F.softmax(logits, dim=-1)
                
                # 2. Safe Entropy Calculation (add epsilon)
                entr = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                layer_entropies.append(entr)
                
            
            # # Stack: [Batch, Layers, Seq]
            stacked_entropies = torch.stack(layer_entropies, dim=1)
            
            # 3. Find Best Layer for each token position
            best_vals, best_layer_indices = torch.min(stacked_entropies, dim=1) # [5, 3, ...]

        # --- B. EXECUTION PHASE (WITH GRADIENTS) ---
        # We use the INDICES found above to gather tensors from the LIVE graph.
        # This creates a valid path for backpropagation.
        
        stacked_hidden_states = torch.stack(outputs.hidden_states, dim=1)
    
        # Prepare indices for the gather operation
        B, S = best_layer_indices.shape
        H = stacked_hidden_states .shape[-1]
        gather_indices = best_layer_indices.view(B, 1, S, 1).expand(-1, -1, -1, H)
        
        # CRITICAL STEP: The Gradient Bridge
        # This function says: "Output = The specific tensor at these indices."
        # Gradients will flow ONLY to the selected indices.
        best_hidden_states = torch.gather(stacked_hidden_states, 1, gather_indices).squeeze(1)

        # print(f"best_hidden_states.requires_grad: {best_hidden_states.requires_grad}\ngather_indices.requires_grad: {gather_indices.requires_grad}\nbest_layer_indices.requires_grad: {best_layer_indices.requires_grad}\nstacked_hidden_states.requires_grad: {stacked_hidden_states.requires_grad}")
        # 3. NORMALIZE & COMPRESS
        best_hidden_states = self.base_model.model.language_model.norm(best_hidden_states)
        return best_hidden_states
    # vishwa
    def forward(
        self,
        class_labels=None,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        token_type_ids=None,
        cache_position=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
        **kwargs,
    ):
        # print("I am used again:)")
        if self.strategy not in ['freeze', 'classifier_only']:
            outputs = self.base_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )
        elif self.strategy == 'freeze':
            with torch.no_grad():
                outputs = self.base_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                labels=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )
        else:
            outputs = self.base_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                labels=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )
        # print(self.use_entropy_minimization)
        # --- LOGIC BRANCHING ---
        if not hasattr(self, 'use_entropy_minimization') or not self.use_entropy_minimization:
            
            # STAGE 1: Standard Last Layer
            best_hidden_states = outputs.hidden_states[-1]
            best_hidden_states = self.base_model.model.language_model.norm(best_hidden_states)
        else:
            # STAGE 2: Top-K Entropy Minimization (Hybrid Strategy)
            best_hidden_states = self.confident_embedding_per_token_extraction(outputs=outputs)
            # We calculate WHICH tokens to pick. Gradients are NOT tracked here.
            # This prevents Memory Explosion and NaNs from the Entropy Math.
        total_loss = 0.0
        if self.training_mode:
            # --- COMMON OUTPUT BLOCK ---
            compressed_input = best_hidden_states.permute(0, 2, 1)
            # Use seqlen-Compression (seqlen -> 1)
            compressed_output = self.seq_compression(compressed_input)
            # Result: [Batch, Hidden, 1] -> [Batch, Hidden]
            final_embedding = compressed_output.squeeze(-1)
            
            class_logits = self.label_classifier(final_embedding)
            if self.strategy not in ['freeze', 'classifier_only']:
                if labels is not None:
                    lm_loss = self.base_model.loss_function(
                        logits=logits,
                        labels=labels,
                        vocab_size=self.base_model.model.get_decoder().config.get_text_config().vocab_size,
                        **kwargs,
                    )
                    total_loss = total_loss + lm_loss
                

            if class_labels is not None:
                if self.criterion_type == "focal":
                    class_loss = focal_loss(class_logits, class_labels, gamma=self.focal_gamma)
                else:
                    class_loss = self.ce_criterion(class_logits, class_labels)
                total_loss = total_loss + class_loss
            
        hidden_states = outputs.hidden_states[-1]
        if self.lm_use:
            logits = self.base_model.lm_head(best_hidden_states)
        else:
            logits = self.base_model.lm_head(hidden_states)
    
        return PaliGemmaCausalLMOutputWithPast(
            loss=total_loss,
            logits=logits, 
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )
        
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # You pass the command down to the real heavy lifter
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self.base_model.enable_input_require_grads()
        self.label_classifier.train()
    
    
    def save_pretrained(self, save_directory):
        self.base_model.save_pretrained(save_directory)
        torch.save(self.label_classifier.state_dict(), f"{save_directory}/label_classifier.pt")
        # Save BOTH compression paths
        torch.save(self.seq_compression.state_dict(), f"{save_directory}/least_random_token.pt")
        # torch.save(self.k_compression.state_dict(), f"{save_directory}/k_compression.pt")

    @classmethod
    def from_pretrained(cls, load_directory, model_name="google/paligemma2-10b-pt-224", num_classes=4, max_seq_len=2048, token_k=50, focal_gamma=2.0):
        # 1. Init
        model = cls(model_name=model_name, num_classes=num_classes, max_seq_len=max_seq_len, token_k=token_k, focal_gamma=focal_gamma)
        
        # 2. Load LoRA
        try:
            model.base_model.load_adapter(load_directory, adapter_name="default")
            print("Successfully loaded LoRA adapters.")
        except: 
            print("Note: No LoRA adapters found.") 
            raise
        # 3. Load Custom Layers
        def load_layer(layer, filename):
            path = os.path.join(load_directory, filename)
            if not os.path.exists(path):
                try: path = hf_hub_download(repo_id=load_directory, filename=filename)
                except: return
            
            state_dict = torch.load(path, map_location="cpu")
            try: layer.load_state_dict(state_dict)
            except RuntimeError: layer.load_state_dict(state_dict, strict=False)
            
            # Force dtype cast
            layer.to(torch.float16)

        load_layer(model.label_classifier, "label_classifier.pt")
        load_layer(model.seq_compression, "least_random_token.pt") # Path 1
        # load_layer(model.k_compression, "k_compression.pt")        # Path 2

        return model

    def __getattr__(self, name):
        """
        Fallback: if attribute not found on wrapper,
        delegate to base_model automatically.
        Covers all internal helpers like:
        _get_initial_cache_position, _update_model_kwargs_for_generation,
        _reorder_cache, config, generation_config, etc.
        """
        try:
            # First try nn.Module's own __getattr__
            return super().__getattr__(name)
        except AttributeError:
            # Then delegate to base_model
            return getattr(self.base_model, name)
    @torch.no_grad()
    def generate(
        self,
        inputs = None,
        generation_config = None,
        logits_processor = None,
        stopping_criteria = None,
        prefix_allowed_tokens_fn = None,
        synced_gpus = None,
        assistant_model = None,
        streamer = None,
        negative_prompt_ids = None,
        negative_prompt_attention_mask = None,
        use_model_defaults = None,
        custom_generate = None,
        **kwargs,
    ) :
        # 1. Handle kwargs, `generation_config`, validate them and obtain generation mode
        generation_mode_kwargs = self.base_model._extract_generation_mode_kwargs(
            custom_generate,
            kwargs,
            synced_gpus,
            assistant_model,
            streamer,
        )

        generation_config, model_kwargs = self.base_model._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        generation_mode = generation_config.get_generation_mode(assistant_model)
        decoding_method = getattr(self, GENERATION_MODES_MAPPING[generation_mode])

        self.base_model._validate_model_kwargs(model_kwargs.copy())
        self.base_model._validate_generation_mode(generation_mode, generation_config, generation_mode_kwargs)
        # print("Decoding method:", decoding_method)
        # print("Bound to object:", decoding_method.__self__)
        # print("Class:", decoding_method.__self__.__class__)
        
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        inputs_tensor, model_input_name, model_kwargs = self.base_model._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        device = inputs_tensor.device
        self.base_model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        input_ids, model_kwargs = self.base_model._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=max(generation_config.num_beams, generation_config.num_return_sequences),
            is_encoder_decoder=self.base_model.config.is_encoder_decoder,
            **model_kwargs,
        )

        input_ids_length = input_ids.shape[1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self.base_model._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )
   
        model_kwargs["logits_to_keep"] = 1

        self.base_model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        prepared_logits_processor = self.base_model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self.base_model._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            tokenizer=generation_mode_kwargs.get("tokenizer"),
        )

        model_kwargs["use_cache"] = generation_config.use_cache
        result = decoding_method(
            # self,
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            **generation_mode_kwargs,
            **model_kwargs,
        )
        return result

    def inference(self, dataset, label_map):
        results = []
        for index, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Running Inference"):
            img_path = row['img']
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] Skipping index {index}: failed to load image ({e})")
                results.append({
                    "index": index,
                    "true_label": row.get("label", None),
                    "pred_label": None,
                    "pred_label_name": None,
                    "probabilities": None,
                    'generated_tags':None
                })
                continue

            text = (
                "Generate tags by considering image, ocr and title \n"
                f"Ocr text associated with the image: {row.get('ocr', '')}\n"
                f"Title associated with the image: {row.get('title', '')}\n"
            )
            if index == 0:
                print(text)
            try:
                label_id, probs, best_layer_indices = self.classify(image, text)
                pred_label_name = label_map.get(label_id, str(label_id))
                print(f"{pred_label_name}\n")
            except Exception as e:
                print(f"[ERROR] Classification failed for index {index}: {e}")
                label_id, pred_label_name, probs = None, None, None
            
            # # try:            
            # inputs = self.processor(
            #     text=text,
            #     images=image,
            #     truncation=True,
            #     padding="max_length",  
            #     max_length=self.max_seq_len,
            #     return_tensors="pt"
            # ).to('cuda')

            # with torch.no_grad():
            #     generated_ids = self.generate(
            #         **inputs,
            #         max_new_tokens=150,
            #         pad_token_id=self.processor.tokenizer.pad_token_id
            #     )
            #     generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            #     ocr = f'Title associated with the image: {row.get("title", "")}'
            #     generated_texts = [generated_texts[0].split(ocr)[-1].strip()]    
            #     print(generated_texts)

            # except Exception as e:
            generated_texts = None
            # print(generated_texts)
            # print(f"Error processing batch {index}-{index}: {e}")
            results.append({
                "index": index,
                "true_label": row.get("label", None),
                "pred_label": label_id,
                "pred_label_name": pred_label_name,
                "probabilities": probs.tolist() if probs is not None else None,
                'generated_tags' : generated_texts if generated_texts is not None else None,
                 'best_layer_indices': best_layer_indices.detach().cpu().tolist()
            })
            
        return results
    
    def classify(self, image, text):
        self.eval()
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            img_token = self.processor.tokenizer.additional_special_tokens[0]
            full_prompt = f"{img_token}{text}"
            
            inputs = self.processor(
                text=full_prompt, images=image, return_tensors="pt",
                padding="max_length", max_length=self.max_seq_len, truncation=True,
            ).to(self.device)

            outputs = self.base_model(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                attention_mask=inputs["attention_mask"], output_hidden_states=True, return_dict=True,
                output_attentions=False 
            )


            layer_entropies = []
            
            for layer_h in outputs.hidden_states:
                
                # last layer -> norm 
                normed = self.base_model.model.language_model.norm(layer_h) # [Batch, Seq, Hidden]
                logits = self.base_model.lm_head(normed).float()
                # 1. Cast to Float32 to prevent NaNs in Softmax
                probs = F.softmax(logits, dim=-1)
                
                # 2. Safe Entropy Calculation (add epsilon)
                entr = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                layer_entropies.append(entr)
                
            
            # # Stack: [Batch, Layers, Seq]
            stacked_entropies = torch.stack(layer_entropies, dim=1)
            
            # 3. Find Best Layer for each token position
            best_vals, best_layer_indices = torch.min(stacked_entropies, dim=1) # [5, 3, ...]
            
            # --- B. EXECUTION PHASE (WITH GRADIENTS) ---
            # We use the INDICES found above to gather tensors from the LIVE graph.
            # This creates a valid path for backpropagation.
            
            stacked_hidden_states = torch.stack(outputs.hidden_states, dim=1)
        
            # Prepare indices for the gather operation
            B, S = best_layer_indices.shape
            H = stacked_hidden_states .shape[-1]
            gather_indices = best_layer_indices.view(B, 1, S, 1).expand(-1, -1, -1, H)
            
            # CRITICAL STEP: The Gradient Bridge
            # This function says: "Output = The specific tensor at these indices."
            # Gradients will flow ONLY to the selected indices.
            best_hidden_states = torch.gather(stacked_hidden_states, 1, gather_indices).squeeze(1)

            # 3. NORMALIZE & COMPRESS
            best_hidden_states = self.base_model.model.language_model.norm(best_hidden_states)
            compressed_input = best_hidden_states.permute(0, 2, 1)
            
            # Use seqlen-Compression (seqlen -> 1)
            self.seq_compression.to(compressed_input.dtype).to(self.device)
            self.label_classifier.to(compressed_input.dtype).to(self.device)

            compressed_output = self.seq_compression(compressed_input)

            final_embedding = compressed_output.squeeze(-1)
            class_logits = self.label_classifier(final_embedding)
            probs = F.softmax(class_logits, dim=-1)
            pred_label = torch.argmax(probs, dim=-1).item()

        return pred_label, probs.cpu().numpy(), best_layer_indices
    
    
    



class MultiTaskingLlavaModel(nn.Module):
    def __init__(self, class_counts = None, num_classes=4, model_name="llava-hf/llava-1.5-7b-hf", train_mode = False,
                 max_seq_len=2048, focal_gamma=2.0, criterion_type="weighted_ce", device_map="auto", strategy='freeze', lm_use=True):
        
        super().__init__()
        '''
        if lm_use == True
        anissssshhhhhh/llava_effective_layer_generation__finetune_toxic_tags
        '''
        self.train_mode = train_mode
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lm_use = lm_use
        self.focal_gamma = focal_gamma
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.criterion_type = criterion_type
        self.strategy = strategy
       
        lora_config = LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )
        self.base_model = LlavaForConditionalGeneration.from_pretrained(
            model_name, quantization_config=bnb_config, device_map=device_map
        )
        self.final_norm = self.base_model.model.language_model.norm

        if class_counts !=None:
            total_samples = class_counts.sum()
            class_weights = total_samples / (len(class_counts) * class_counts)

            self.register_buffer("class_weights", class_weights)
            self.ce_criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        self.base_model = get_peft_model(self.base_model, lora_config)
        if self.strategy=='freeze':
            for param in self.base_model.parameters():
                param.requires_grad = False
        self.hidden_size = self.base_model.config.text_config.hidden_size
        for mode_method in GENERATION_MODES_MAPPING.values():
            method = getattr(self.base_model, mode_method, None)
            if method is not None:
                setattr(self, mode_method, types.MethodType(method.__func__, self))
        # self.final_norm = self.base_model.model.language_model.norm   
    
        # YOUR ARCHITECTURE: (2048 -> 1024 -> 1)
        self.seq_compression = nn.Sequential(
            nn.Linear(self.max_seq_len, 1024), # max_seq_len, 1024  - (0, 1 - 0), (2, 3) - 1
            nn.GELU(),
            nn.Linear(1024, 1) # 1/2048
        )

        self.label_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.LayerNorm(self.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size // 4, num_classes),
        )      
    def __getattr__(self, name):
        """
        Fallback: if attribute not found on wrapper,
        delegate to base_model automatically.
        Covers all internal helpers like:
        _get_initial_cache_position, _update_model_kwargs_for_generation,
        _reorder_cache, config, generation_config, etc.
        """
        try:
            # First try nn.Module's own __getattr__
            return super().__getattr__(name)
        except AttributeError:
            # Then delegate to base_model
            return getattr(self.base_model, name)  
    def confident_embedding_per_token_extraction(self, outputs):
        with torch.no_grad():
            layer_entropies = []
            
            for layer_h in outputs.hidden_states:
                
                # last layer -> norm 
                normed = self.base_model.model.language_model.norm(layer_h) # [Batch, Seq, Hidden]
                
                # 1. Cast to Float32 to prevent NaNs in Softmax
                logits = self.base_model.lm_head(normed).float()  # batch, seq, hiddensize, [hiddensize, vocab]  
                probs = F.softmax(logits, dim=-1)
                
                # 2. Safe Entropy Calculation (add epsilon)
                entr = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                layer_entropies.append(entr)
                
            
            # # Stack: [Batch, Layers, Seq]
            stacked_entropies = torch.stack(layer_entropies, dim=1)
            
            # 3. Find Best Layer for each token position
            best_vals, best_layer_indices = torch.min(stacked_entropies, dim=1) # [5, 3, ...]

        # --- B. EXECUTION PHASE (WITH GRADIENTS) ---
        # We use the INDICES found above to gather tensors from the LIVE graph.
        # This creates a valid path for backpropagation.
        
        stacked_hidden_states = torch.stack(outputs.hidden_states, dim=1)
    
        # Prepare indices for the gather operation
        B, S = best_layer_indices.shape
        H = stacked_hidden_states .shape[-1]
        gather_indices = best_layer_indices.view(B, 1, S, 1).expand(-1, -1, -1, H)
        
        # CRITICAL STEP: The Gradient Bridge
        # This function says: "Output = The specific tensor at these indices."
        # Gradients will flow ONLY to the selected indices.
        best_hidden_states = torch.gather(stacked_hidden_states, 1, gather_indices).squeeze(1)

        # print(f"best_hidden_states.requires_grad: {best_hidden_states.requires_grad}\ngather_indices.requires_grad: {gather_indices.requires_grad}\nbest_layer_indices.requires_grad: {best_layer_indices.requires_grad}\nstacked_hidden_states.requires_grad: {stacked_hidden_states.requires_grad}")
        # 3. NORMALIZE & COMPRESS
        best_hidden_states = self.base_model.model.language_model.norm(best_hidden_states)
        return best_hidden_states
    def collate_fn(self, examples):
        images = []
        texts = []
        class_labels = [example[3] for example in examples]

        for example in examples:
            image = example[0]
            ground_truth = str(example[2])
            text = example[1]
            
            # Ensure image is RGB (LLaVA 1.5 is sensitive to this)
            if image.mode != "RGB":
                image = image.convert("RGB")
            images.append(image)

            # LLaVA 1.5 Prompt Format
            # Note: Some versions of LlavaProcessor use "USER: <image>\n{text} ASSISTANT:"
            # ground_truth = str(example[2]) + processor.tokenizer.eos_token
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ground_truth}],
                }
            ]
            
            # tokenize=False because processor() will handle tokenization below
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=False)
            texts.append(text_prompt)

        # LLaVA 1.5 Processor
        batch = self.processor(
            text=texts, 
            images=images, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_seq_len, 
            return_tensors="pt"
        )

        # Standard Label Prep
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        batch["class_labels"] = torch.tensor(class_labels, dtype=torch.long)

        return batch
    # vishwa
    def forward(
            self,
        input_ids = None,
        pixel_values = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        vision_feature_layer = None,
        vision_feature_select_strategy = None,
        labels = None,
        cache_position  = None,
        logits_to_keep = 0,
        image_sizes = None,
        class_labels=None,
        **kwargs,
        ):
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.base_model.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.base_model.config.vision_feature_select_strategy
        )
       
        if self.strategy not in ['freeze', 'classifier_only']: ## freeze and classifier only classification loss is used
            outputs = self.base_model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        vision_feature_layer=vision_feature_layer,
                        vision_feature_select_strategy=vision_feature_select_strategy,
                        cache_position=cache_position,
                        image_sizes=image_sizes,
                        output_hidden_states=True,
                        **kwargs,
                    )

        elif self.strategy == 'freeze':
            with torch.no_grad():
                outputs = self.base_model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        vision_feature_layer=vision_feature_layer,
                        vision_feature_select_strategy=vision_feature_select_strategy,
                        cache_position=cache_position,
                        image_sizes=image_sizes,
                        output_hidden_states=True,

                        **kwargs,
                    )
        else:
            outputs = self.base_model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        vision_feature_layer=vision_feature_layer,
                        vision_feature_select_strategy=vision_feature_select_strategy,
                        cache_position=cache_position,
                        image_sizes=image_sizes,
                        output_hidden_states=True,

                        **kwargs,
                    )
            
            
        total_loss = 0.0        

        # layers are in outputs.hidden_states
        if not hasattr(self, 'use_entropy_minimization') or not self.use_entropy_minimization:
            print("I am usable")
            best_hidden_states = outputs.hidden_states[-1]
            best_hidden_states =  self.final_norm(best_hidden_states)
            
            # Pad or truncate to max_seq_len to match linear layer
            current_len = best_hidden_states.shape[1]
            if current_len < self.max_seq_len:
                best_hidden_states = F.pad(best_hidden_states, (0, 0, 0, self.max_seq_len - current_len))
            else:
                best_hidden_states = best_hidden_states[:, :self.max_seq_len, :]
        else:
            # STAGE 2: Entropy Minimization
            # a, b, c
            ## 3, hidden_dimension
             ## vocabulary, hidden_dimension
            best_hidden_states = self.confident_embedding_per_token_extraction(outputs=outputs)
        if self.train_mode:
            compressed_input = best_hidden_states.permute(0, 2, 1)
            compressed_output = self.seq_compression(compressed_input)

            final_embedding = compressed_output.squeeze(-1)
            class_logits = self.label_classifier(final_embedding)
            if self.strategy not in ['freeze', 'classifier_only']:
                if labels is not None:

                    lm_loss = self.base_model.loss_function(
                        logits=logits,
                        labels=labels,
                        vocab_size=self.base_model.model.get_decoder().config.get_text_config().vocab_size,
                        **kwargs,
                    )

                    total_loss = total_loss + lm_loss
                if class_labels is not None:
                    if self.criterion_type == "focal":
                        class_loss = focal_loss(class_logits, class_labels, gamma=self.focal_gamma)
                    else:
                        class_loss = self.ce_criterion(class_logits, class_labels)
                    total_loss = total_loss + class_loss

        hidden_states = outputs.hidden_states[-1]
        if self.lm_use:
            # print(self.lm_use)
            logits = self.base_model.lm_head(best_hidden_states)
        else:
            logits = self.base_model.lm_head(hidden_states)
        

        
        return LlavaCausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )
    
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # You pass the command down to the real heavy lifter
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self.base_model.enable_input_require_grads()
        self.label_classifier.train()
    
    
    def save_pretrained(self, save_directory):
        # Qwen2-VL specific save path
        self.base_model.save_pretrained(save_directory)
        torch.save(self.label_classifier.state_dict(), f"{save_directory}/label_classifier.pt")
        torch.save(self.seq_compression.state_dict(), f"{save_directory}/least_random_token.pt")

    @classmethod
    def from_pretrained(cls, load_directory, model_name="llava-hf/llava-1.5-7b-hf", num_classes=4, max_seq_len=2048, token_k=50, focal_gamma=2.0):
        # 1. Init with Qwen model name
        model = cls(model_name=model_name, num_classes=num_classes, max_seq_len=max_seq_len, focal_gamma=focal_gamma)
        
        # 2. Load LoRA
        try:
            model.base_model.load_adapter(load_directory, adapter_name="default")
            print("Successfully loaded LoRA adapters.")
        except Exception as e: 
            print(f"Note: No LoRA adapters found or error loading: {e}")
            raise

        # 3. Load Custom Layers
        def load_layer(layer, filename):
            path = os.path.join(load_directory, filename)
            if not os.path.exists(path):
                try: 
                    path = hf_hub_download(repo_id=load_directory, filename=filename)
                except: 
                    return
            
            state_dict = torch.load(path, map_location="cpu")
            try: 
                layer.load_state_dict(state_dict)
            except RuntimeError: 
                layer.load_state_dict(state_dict, strict=False)
            
            # Match compute dtype (usually float16/bfloat16 for Qwen)
            layer.to(torch.float16)

        load_layer(model.label_classifier, "label_classifier.pt")
        load_layer(model.seq_compression, "least_random_token.pt")

        return model
    @torch.no_grad()
    def generate(
        self,
        inputs = None,
        generation_config = None,
        logits_processor = None,
        stopping_criteria = None,
        prefix_allowed_tokens_fn = None,
        synced_gpus = None,
        assistant_model = None,
        streamer = None,
        negative_prompt_ids = None,
        negative_prompt_attention_mask = None,
        use_model_defaults = None,
        custom_generate = None,
        **kwargs,
    ) :
        # 1. Handle kwargs, `generation_config`, validate them and obtain generation mode
        generation_mode_kwargs = self.base_model._extract_generation_mode_kwargs(
            custom_generate,
            kwargs,
            synced_gpus,
            assistant_model,
            streamer,
        )

        generation_config, model_kwargs = self.base_model._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        generation_mode = generation_config.get_generation_mode(assistant_model)
        decoding_method = getattr(self, GENERATION_MODES_MAPPING[generation_mode])

        self.base_model._validate_model_kwargs(model_kwargs.copy())
        self.base_model._validate_generation_mode(generation_mode, generation_config, generation_mode_kwargs)
        # print("Decoding method:", decoding_method)
        # print("Bound to object:", decoding_method.__self__)
        # print("Class:", decoding_method.__self__.__class__)
        
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        inputs_tensor, model_input_name, model_kwargs = self.base_model._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        device = inputs_tensor.device
        self.base_model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        input_ids, model_kwargs = self.base_model._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=max(generation_config.num_beams, generation_config.num_return_sequences),
            is_encoder_decoder=self.base_model.config.is_encoder_decoder,
            **model_kwargs,
        )

        input_ids_length = input_ids.shape[1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self.base_model._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )
   
        model_kwargs["logits_to_keep"] = 1

        self.base_model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        prepared_logits_processor = self.base_model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self.base_model._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            tokenizer=generation_mode_kwargs.get("tokenizer"),
        )

        model_kwargs["use_cache"] = generation_config.use_cache
        result = decoding_method(
            # self,
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            **generation_mode_kwargs,
            **model_kwargs,
        )
        return result
    def inference(self, dataset, label_map):
        results = []
        for index, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Running Inference"):
            img_path = row['img']
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] Skipping index {index}: failed to load image ({e})")
                results.append({
                    "index": index,
                    "true_label": row.get("label", None),
                    "pred_label": None,
                    "pred_label_name": None,
                    "probabilities": None,
                    'generated_tags':None
                })
                continue

            text = (
                "Generate tags by considering image, ocr and title \n"
                f"Ocr text associated with the image: {row.get('ocr', '')}\n"
                f"Title associated with the image: {row.get('title', '')}\n"
            )
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"{text}"},
                    ],
                },
            ]   
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True) 
              
            if index == 0:
                print(text)
            try:
                label_id, probs, best_layer_indices = self.classify(image, text)
                pred_label_name = label_map.get(label_id, str(label_id))
                print(f"{pred_label_name}\n")
            except Exception as e:
                print(f"[ERROR] Classification failed for index {index}: {e}")
                label_id, pred_label_name, probs = None, None, None
            
            # # try:            
            # inputs = self.processor(
            #     text=[prompt], 
            #     images=[image], 
            #     return_tensors="pt",
            #     padding="max_length", 
            #     max_length=2048, 
            #     truncation=True,
            # ).to('cuda')

            # with torch.no_grad():
            #     generated_ids = self.generate(
            #         **inputs,
            #         max_new_tokens=150,
            #         repetition_penalty=1.5,
            #         eos_token_id=self.processor.tokenizer.eos_token_id,
            #         do_sample=False,
            #         pad_token_id=self.processor.tokenizer.pad_token_id
            #     )
            #     generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            #     generated_texts =   [
            #         text.lower().split("assistant:")[-1].strip()
            #         for text in generated_texts
            #     ]   
            #     print(generated_texts)
            # except Exception as e:
            generated_texts = None
            #     print(generated_texts)
            #     print(f"Error processing batch {index}-{index}: {e}")
            results.append({
                "index": index,
                "true_label": row.get("label", None),
                "pred_label": label_id,
                "pred_label_name": pred_label_name,
                "probabilities": probs.tolist() if probs is not None else None,
                'generated_tags' : generated_texts if generated_texts is not None else None,
                'best_layer_indices': best_layer_indices.detach().cpu().tolist()

                
            })
            
        return results
    def classify(self, image, text):
        self.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            # 1. Prepare the LLaVA 1.5 prompt format
            # Standard: "USER: <image>\n{text} ASSISTANT:"
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"{text}"},
                    ],
                },
            ]   
            # add_generation_prompt=True adds the "ASSISTANT:" suffix
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)         
            
            # 2. Process inputs (No image_sizes needed for LLaVA 1.5)
            # Ensure image is RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            inputs = self.processor(
                text=[prompt], 
                images=[image], 
                return_tensors="pt",
                padding="max_length", 
                max_length=self.max_seq_len, 
                truncation=True,
            ).to(device)

            # 3. Model Forward Pass
            # LLaVA 1.5 expects input_ids, pixel_values, and attention_mask
            outputs = self.base_model(
                input_ids=inputs["input_ids"], 
                pixel_values=inputs["pixel_values"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True, 
                return_dict=True,
            )

            # 4. Hidden State Selection
            if not hasattr(self, 'use_entropy_minimization') or not self.use_entropy_minimization:
                # Standard path: Use last hidden layer
                best_hidden_states = outputs.hidden_states[-1]
            else:
                # Entropy path: Select best layers per token
                layer_entropies = []
                for layer_h in outputs.hidden_states:
                    normed = self.final_norm(layer_h)
                    logits = self.base_model.lm_head(normed).float()
                    probs = F.softmax(logits, dim=-1)
                    entr = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                    layer_entropies.append(entr)
                
                stacked_entropies = torch.stack(layer_entropies, dim=1)
                _, best_layer_indices = torch.min(stacked_entropies, dim=1)

                stacked_hidden_states = torch.stack(outputs.hidden_states, dim=1)
                B, L, S, H = stacked_hidden_states.shape
                
                gather_indices = best_layer_indices.view(B, 1, S, 1).expand(-1, -1, -1, H)
                best_hidden_states = torch.gather(stacked_hidden_states, 1, gather_indices).squeeze(1)

            # 5. Normalization and Sequence Compression
            # Apply the backbone's final LayerNorm
            final_hidden = self.final_norm(best_hidden_states)
            
            # Pad/Truncate the sequence to exactly match self.max_seq_len (2048)
            current_len = final_hidden.shape[1]
            if current_len < self.max_seq_len:
                final_hidden = F.pad(final_hidden, (0, 0, 0, self.max_seq_len - current_len))
            else:
                final_hidden = final_hidden[:, :self.max_seq_len, :]

            # Permute to [Batch, Hidden, Seq] for the Linear(2048, 1024) layer
            compressed_input = final_hidden.permute(0, 2, 1)
            
            # Ensure precision matches (float16/bfloat16)
            self.seq_compression.to(compressed_input.dtype).to(device)
            self.label_classifier.to(compressed_input.dtype).to(device)

            # Run Custom Heads
            compressed_output = self.seq_compression(compressed_input) # [Batch, Hidden, 1]
            final_embedding = compressed_output.squeeze(-1)            # [Batch, Hidden]
            class_logits = self.label_classifier(final_embedding)
            
            probs = F.softmax(class_logits, dim=-1)
            pred_label = torch.argmax(probs, dim=-1).item()

        return pred_label, probs.cpu().numpy(), best_layer_indices