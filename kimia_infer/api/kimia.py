import os
import time
import tqdm
import torch
from loguru import logger
from huggingface_hub import cached_assets_path
from transformers import AutoModelForCausalLM
import soundfile as sf
from kimia_infer.models.detokenizer import get_audio_detokenizer
from .prompt_manager import KimiAPromptManager
from kimia_infer.utils.sampler import KimiASampler
from modelscope import snapshot_download
start_time = 0
output_dir = "my_test/"
"""
想看却暂时没找到源码的：
from kimia_infer.utils.sampler import KimiASampler
到底KimiASampler里面写了什么，是用来干什么的


"""

class KimiAudio(object):
    def __init__(self, model_path: str, load_detokenizer: bool = True):
        logger.info(f"Loading kimi-audio main model")

        if os.path.exists(model_path):
            # local path
            cache_path = model_path
        else:
            # cache everything if model_path is a model-id
            cache_path = snapshot_download(model_path)
    
        logger.info(f"Looking for resources in {cache_path}")
        logger.info(f"Loading whisper model")
        self.alm = AutoModelForCausalLM.from_pretrained(
            cache_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self.alm = self.alm.to(torch.cuda.current_device())

        model_config = self.alm.config
        self.kimia_text_audiodelaytokens = model_config.kimia_mimo_audiodelaytokens
        self.kimia_token_offset = model_config.kimia_token_offset

        self.prompt_manager = KimiAPromptManager(
            model_path=cache_path, kimia_token_offset=self.kimia_token_offset, kimia_text_audiodelaytokens=self.kimia_text_audiodelaytokens
        )

        if load_detokenizer:
            logger.info(f"Loading detokenizer")
            # need to compile extension moudules for the first time, it may take several minutes.
            self.detokenizer = get_audio_detokenizer(cache_path)
        else:
            # in this case, you're not allowed to generate audio(wav)
            self.detokenizer = None

        self.extra_tokens = self.prompt_manager.extra_tokens
        self.eod_ids = [self.extra_tokens.msg_end, self.extra_tokens.media_end]

    @torch.inference_mode()
    def _generate_loop(#生成回复,max_new_tokens = 80
        self,
        audio_input_ids: torch.Tensor,  # input audio tokens
        text_input_ids: torch.Tensor = None,  # input text tokens if use multi-input
        max_new_tokens: int = 50,
        audio_top_k: int = 5,
        audio_temperature: float = 0.0,
        audio_repetition_penalty: float = 1.0,
        audio_repetition_window_size: int = 64,
        text_top_k: int = 5,
        text_temperature: float = 0.0,
        text_repetition_penalty: float = 1.0,
        text_repetition_window_size: int = 16,
        is_continuous_mask: torch.Tensor = None,
        continous_feature: torch.Tensor = None,
        output_type: str = "text",
        number_of_tokens_to_a_wav = 30,
    ):
        sampler = KimiASampler(
            audio_top_k=audio_top_k,
            audio_temperature=audio_temperature,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
        )
        text_stream_is_finished = False
        previous_audio_tokens = torch.zeros(#生成一张张量大小为0，形状为4096的矩阵
            (4096,),
            dtype=torch.int,
            device=torch.cuda.current_device(),
        )
        text_previous_tokens = torch.zeros(
            (4096,),
            dtype=torch.int,
            device=torch.cuda.current_device(),
        )

        decoder_input_audio_ids = audio_input_ids.clone()
        decoder_input_text_ids = text_input_ids.clone()
        decoder_position_ids = (
            torch.arange(
                0, decoder_input_audio_ids.shape[1], device=torch.cuda.current_device()
            )
            .unsqueeze(0)
            .long()
        )
        decoder_input_whisper_feature = continous_feature
        decoder_is_continuous_mask = is_continuous_mask
        past_key_values = None

        last_position_id = decoder_input_audio_ids.shape[1] - 1

        valid_text_length = 0
        valid_audio_length = 0
        first_time_flag = True
        temp_start = 0#记录暂时性生成到了哪里
        temp_time = time.time()
        for i in tqdm.tqdm(#生成tokens，长度为max_new_tokens
            range(max_new_tokens), desc="Generating tokens", disable=False
        ):
            audio_logits, text_logits, past_key_values = self.alm.forward(#预训练模型开始推理,采样80次
                input_ids=decoder_input_audio_ids,
                text_input_ids=decoder_input_text_ids,
                whisper_input_feature=decoder_input_whisper_feature,
                is_continuous_mask=decoder_is_continuous_mask,
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                return_dict=False,
            )

            # Sample text token using the sampler
            next_token_text = sampler.sample_text_logits(
                text_logits, recent_tokens=text_previous_tokens[:i] if i > 0 else None
            )
            # Sample audio token using the sampler
            next_audio_token = sampler.sample_audio_logits(
                audio_logits, recent_tokens=previous_audio_tokens[:i] if i > 0 else None
            )
            if text_stream_is_finished:
                next_token_text.fill_(self.extra_tokens.kimia_text_blank)
            elif next_token_text.item() == self.extra_tokens.kimia_text_eos:
                text_stream_is_finished = True
            else:
                valid_text_length += 1

            text_previous_tokens[i : i + 1] = next_token_text

            if i < self.kimia_text_audiodelaytokens:#audio_token无论如何都是正确的，只不过有可能fill
                next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
            else:
                if output_type == "text":
                    next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
                else:
                    valid_audio_length += 1

            previous_audio_tokens[i : i + 1] = next_audio_token
            audio_stream_is_finished = next_audio_token.item() in self.eod_ids
            ### my code ###
            if ((i%number_of_tokens_to_a_wav==0 and i !=0 ) or audio_stream_is_finished):#判断当audio生成完毕时或token数目到了该生成一次wav文件的时候
                #过滤tokens，把没用的删除
                generated_wav_tokens = [t for t in previous_audio_tokens[temp_start:i] if t >= self.kimia_token_offset]
                generated_wav_tokens = torch.tensor(generated_wav_tokens).unsqueeze(0)
                generated_wav_tokens = generated_wav_tokens - self.kimia_token_offset
                #生成audio
                generated_wav = self.detokenize_audio(generated_wav_tokens)
                if(not os.path.exists(output_dir+str(number_of_tokens_to_a_wav))):#生成的文件夹
                    os.mkdir(output_dir+str(number_of_tokens_to_a_wav))
                if(not os.path.exists(output_dir+str(number_of_tokens_to_a_wav)+"/output")):
                    os.mkdir(output_dir+str(number_of_tokens_to_a_wav)+"/output")
                sf.write(
                        os.path.join(output_dir+str(number_of_tokens_to_a_wav)+"/output", "hit_stream_output_"+str(i)+".wav"),
                        generated_wav.detach().cpu().view(-1).numpy(),
                        24000,
                        )
                temp_start = i
                if(first_time_flag):#记录是否是第一次生成，如果是需要记录从投入视频到第一个token产出的时间
                    temp_time = time.time()
                    print("-"*20)
                    print("from start to end:",start_time - temp_time)
                    print("-"*20)
                    first_time_flag = False
                else:#记录当前token产生的时间
                    print("-"*20)
                    now_time = time.time()
                    print(str(i)+" audio cost time:",now_time-temp_time)
                    temp_time = now_time
                    print("-"*20)
            ### my code ###
            

            if (
                output_type == "text"
                and text_stream_is_finished
                or output_type == "both"
                and audio_stream_is_finished
            ):
                return_text_tokens = (
                    text_previous_tokens[:valid_text_length]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                return_audio_tokens = (
                    previous_audio_tokens[
                        self.kimia_text_audiodelaytokens : valid_audio_length
                        + self.kimia_text_audiodelaytokens
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                return return_audio_tokens, return_text_tokens
            else:
                decoder_input_audio_ids = next_audio_token.unsqueeze(1)
                decoder_input_text_ids = next_token_text.unsqueeze(1)

                decoder_position_ids = (
                    torch.zeros(1, 1, device=torch.cuda.current_device())
                    .fill_(last_position_id + 1)
                    .long()
                    .view(1, 1)
                )
                last_position_id += 1

                decoder_input_whisper_feature = None
                decoder_is_continuous_mask = None
        return_text_tokens = (
            text_previous_tokens[:valid_text_length].detach().cpu().numpy().tolist()
        )
        return_audio_tokens = (
            previous_audio_tokens[
                self.kimia_text_audiodelaytokens : valid_audio_length
                + self.kimia_text_audiodelaytokens
            ]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
        return return_audio_tokens, return_text_tokens

    @torch.inference_mode()
    def generate(
        self,
        chats: list[dict],
        output_type="text",
        audio_temperature=0.0,
        audio_top_k=5,
        text_temperature=0.0,
        text_top_k=5,
        audio_repetition_penalty=1.0,
        audio_repetition_window_size=64,
        text_repetition_penalty=1.0,
        text_repetition_window_size=16,
        max_new_tokens=-1,
        number_of_tokens_to_a_wav=30,
    ):
        ## TODO: 需要一个check函数，检查输入的history格式是否合法
        ## 比如，对于ASR任务，一定是: text-instruction/audio-instruction + audio-content, 我理解content和instruction是不能换位置的
        ## assistant前必须有user等等，我觉得最好做一下check
        global start_time 
        start_time = time.time()
        assert output_type in ["text", "both"]
        start_get_prompt_time = time.time()
        history = self.prompt_manager.get_prompt(chats, output_type=output_type)#从chats里面拿到大模型的prompt
        end_get_prompt_time = time.time()
        print("-"*20)
        print("time used from audio to prompt:",end_get_prompt_time-start_get_prompt_time)
        print("-"*20)
        audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()#从prompt里面拿到要的信息
        audio_features = history.continuous_feature
        generated_wav_tokens = []
        generated_text_tokens = []

        if output_type == "both":
            max_new_tokens = int(12.5 * 120) - audio_input_ids.shape[1]
        else:
            if max_new_tokens == -1:
                max_new_tokens = 7500 - audio_input_ids.shape[1]

        audio_input_ids = audio_input_ids.to(torch.cuda.current_device())
        text_input_ids = text_input_ids.to(torch.cuda.current_device())
        is_continuous_mask = is_continuous_mask.to(torch.cuda.current_device())
        audio_features = [f.to(torch.cuda.current_device()) for f in audio_features]
        start_generate_loop = time.time()
        generated_wav_tokens, generated_text_tokens = self._generate_loop(
            audio_input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            max_new_tokens=max_new_tokens,
            audio_temperature=audio_temperature,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
            is_continuous_mask=is_continuous_mask,
            continous_feature=audio_features,
            output_type=output_type,
            number_of_tokens_to_a_wav=number_of_tokens_to_a_wav,
        )
        end_generate_loop = time.time()
        print("-"*20)
        print("time used to generate answer:",end_generate_loop-start_generate_loop)
        print("-"*20)
        start_filter = time.time()
        generated_wav_tokens = [
            t for t in generated_wav_tokens if t >= self.kimia_token_offset
        ]  #  filter out the illegal tokens

        generated_wav_tokens = torch.tensor(generated_wav_tokens).unsqueeze(0)
        generated_wav_tokens = generated_wav_tokens - self.kimia_token_offset
        end_filter = time.time()
        print("-"*20)
        print("time used in filtering:",end_filter-start_filter)
        print("-"*20)
        generated_text_tokens = [t for t in generated_text_tokens if t < self.kimia_token_offset]

        start_detokenize_text = time.time()
        generated_text = self.detokenize_text(generated_text_tokens)
        end_detokenize_text = time.time()
        print("-"*20)
        print("time used in detokenizing text:",end_detokenize_text-start_detokenize_text)
        print("-"*20)
        if self.detokenizer is not None and output_type == "both":
            start_detokenize_audio = time.time()
            generated_wav = self.detokenize_audio(generated_wav_tokens)
            end_detokenize_audio = time.time()
            print("-"*20)
            print("time used in detokenizing audio",end_detokenize_audio-start_detokenize_audio)
            print("-"*20)
        else:
            generated_wav = None

        return generated_wav, generated_text

    def detokenize_audio(self, audio_tokens):
        if self.detokenizer is None:
            raise ValueError("Detokenizer is not initialized")
        self.detokenizer.clear_states()
        chunk_size = 30  # hard-coded right now
        first_chunk_size = 30
        cache_speech_collection = []
        audio_tokens = audio_tokens.to(torch.cuda.current_device())
        audio_tokens = audio_tokens.long()
        num_audio_tokens = audio_tokens.size(1)
        first_chunk_semantic_tokens = audio_tokens[:, :first_chunk_size]
        gen_speech = self.detokenizer.detokenize_streaming(
            first_chunk_semantic_tokens,
            is_final=(num_audio_tokens <= first_chunk_size),
            upsample_factor=4,
        )
        cache_speech_collection.append(gen_speech)

        if num_audio_tokens > first_chunk_size:
            res_semantic_tokens = audio_tokens[:, first_chunk_size:]
            for i in range(0, res_semantic_tokens.size(1), chunk_size):
                chunk_semantic_tokens = res_semantic_tokens[:, i : i + chunk_size]
                gen_speech = self.detokenizer.detokenize_streaming(
                    chunk_semantic_tokens,
                    upsample_factor=4,
                    is_final=(i + chunk_size >= res_semantic_tokens.size(1)),
                )
                cache_speech_collection.append(gen_speech)

        gen_speech = torch.cat(cache_speech_collection, dim=-1)
        return gen_speech

    def detokenize_text(self, text_tokens):
        valid_text_ids = []
        for x in text_tokens:
            if x == self.extra_tokens.kimia_text_eos:
                break
            valid_text_ids.append(x)
        return self.prompt_manager.text_tokenizer.decode(valid_text_ids)
