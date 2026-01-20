from typing import Optional

import torch
import torchvision

from invokeai.backend.flux.text_conditioning import (
    FluxReduxConditioning,
    FluxRegionalTextConditioning,
    FluxTextConditioning,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Range
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.mask import to_standard_float_mask


class RegionalPromptingExtension:
    """A class for managing regional prompting with FLUX.

    This implementation is inspired by https://arxiv.org/pdf/2411.02395 (though there are significant differences).
    """

    def __init__(
        self,
        regional_text_conditioning: FluxRegionalTextConditioning,
        restricted_attn_mask: torch.Tensor | None = None,
    ):
        self.regional_text_conditioning = regional_text_conditioning
        self.restricted_attn_mask = restricted_attn_mask

    def get_double_stream_attn_mask(self, block_index: int) -> torch.Tensor | None:
        order = [self.restricted_attn_mask, None]
        return order[block_index % len(order)]

    def get_single_stream_attn_mask(self, block_index: int) -> torch.Tensor | None:
        order = [self.restricted_attn_mask, None]
        return order[block_index % len(order)]

    @classmethod
    def from_text_conditioning(
        cls,
        text_conditioning: list[FluxTextConditioning],
        redux_conditioning: list[FluxReduxConditioning],
        img_seq_len: int,
    ):
        """Create a RegionalPromptingExtension from a list of text conditionings.

        Args:
            text_conditioning (list[FluxTextConditioning]): The text conditionings to use for regional prompting.
            img_seq_len (int): The image sequence length (i.e. packed_height * packed_width).
        """
        regional_text_conditioning = cls._concat_regional_text_conditioning(text_conditioning, redux_conditioning)
        attn_mask_with_restricted_img_self_attn = cls._prepare_restricted_attn_mask(
            regional_text_conditioning, img_seq_len
        )
        return cls(
            regional_text_conditioning=regional_text_conditioning,
            restricted_attn_mask=attn_mask_with_restricted_img_self_attn,
        )

    # Keeping _prepare_unrestricted_attn_mask for reference as an alternative masking strategy:
    #
    # @classmethod
    # def _prepare_unrestricted_attn_mask(
    #     cls,
    #     regional_text_conditioning: FluxRegionalTextConditioning,
    #     img_seq_len: int,
    # ) -> torch.Tensor:
    #     """Prepare an 'unrestricted' attention mask. In this context, 'unrestricted' means that:
    #     - img self-attention is not masked.
    #     - img regions attend to both txt within their own region and to global prompts.
    #     """
    #     device = TorchDevice.choose_torch_device()

    #     # Infer txt_seq_len from the t5_embeddings tensor.
    #     txt_seq_len = regional_text_conditioning.t5_embeddings.shape[1]

    #     # In the attention blocks, the txt seq and img seq are concatenated and then attention is applied.
    #     # Concatenation happens in the following order: [txt_seq, img_seq].
    #     # There are 4 portions of the attention mask to consider as we prepare it:
    #     # 1. txt attends to itself
    #     # 2. txt attends to corresponding regional img
    #     # 3. regional img attends to corresponding txt
    #     # 4. regional img attends to itself

    #     # Initialize empty attention mask.
    #     regional_attention_mask = torch.zeros(
    #         (txt_seq_len + img_seq_len, txt_seq_len + img_seq_len), device=device, dtype=torch.float16
    #     )

    #     for image_mask, t5_embedding_range in zip(
    #         regional_text_conditioning.image_masks, regional_text_conditioning.t5_embedding_ranges, strict=True
    #     ):
    #         # 1. txt attends to itself
    #         regional_attention_mask[
    #             t5_embedding_range.start : t5_embedding_range.end, t5_embedding_range.start : t5_embedding_range.end
    #         ] = 1.0

    #         # 2. txt attends to corresponding regional img
    #         # Note that we reshape to (1, img_seq_len) to ensure broadcasting works as desired.
    #         fill_value = image_mask.view(1, img_seq_len) if image_mask is not None else 1.0
    #         regional_attention_mask[t5_embedding_range.start : t5_embedding_range.end, txt_seq_len:] = fill_value

    #         # 3. regional img attends to corresponding txt
    #         # Note that we reshape to (img_seq_len, 1) to ensure broadcasting works as desired.
    #         fill_value = image_mask.view(img_seq_len, 1) if image_mask is not None else 1.0
    #         regional_attention_mask[txt_seq_len:, t5_embedding_range.start : t5_embedding_range.end] = fill_value

    #     # 4. regional img attends to itself
    #     # Allow unrestricted img self attention.
    #     regional_attention_mask[txt_seq_len:, txt_seq_len:] = 1.0

    #     # Convert attention mask to boolean.
    #     regional_attention_mask = regional_attention_mask > 0.5

    #     return regional_attention_mask

    @classmethod
    def _prepare_restricted_attn_mask(
        cls,
        regional_text_conditioning: FluxRegionalTextConditioning,
        img_seq_len: int,
    ) -> torch.Tensor | None:
        """Prepare a 'restricted' attention mask. In this context, 'restricted' means that:
        - img self-attention is only allowed within regions.
        - img regions only attend to txt within their own region, not to global prompts.
        """
        # Identify background region. I.e. the region that is not covered by any region masks.
        background_region_mask: None | torch.Tensor = None
        for image_mask in regional_text_conditioning.image_masks:
            if image_mask is not None:
                if background_region_mask is None:
                    background_region_mask = torch.ones_like(image_mask)
                background_region_mask *= 1 - image_mask

        if background_region_mask is None:
            # There are no region masks, short-circuit and return None.
            # TODO(ryand): We could restrict txt-txt attention across multiple global prompts, but this would
            # is a rare use case and would make the logic here significantly more complicated.
            return None

        device = TorchDevice.choose_torch_device()

        # Infer txt_seq_len from the active text embeddings tensor.
        txt_seq_len = regional_text_conditioning.active_text_embeddings().shape[1]

        # In the attention blocks, the txt seq and img seq are concatenated and then attention is applied.
        # Concatenation happens in the following order: [txt_seq, img_seq].
        # There are 4 portions of the attention mask to consider as we prepare it:
        # 1. txt attends to itself
        # 2. txt attends to corresponding regional img
        # 3. regional img attends to corresponding txt
        # 4. regional img attends to itself

        # Initialize empty attention mask.
        regional_attention_mask = torch.zeros(
            (txt_seq_len + img_seq_len, txt_seq_len + img_seq_len), device=device, dtype=torch.float16
        )

        for image_mask, t5_embedding_range in zip(
            regional_text_conditioning.image_masks, regional_text_conditioning.t5_embedding_ranges, strict=True
        ):
            # 1. txt attends to itself
            regional_attention_mask[
                t5_embedding_range.start : t5_embedding_range.end, t5_embedding_range.start : t5_embedding_range.end
            ] = 1.0

            if image_mask is not None:
                # 2. txt attends to corresponding regional img
                # Note that we reshape to (1, img_seq_len) to ensure broadcasting works as desired.
                regional_attention_mask[t5_embedding_range.start : t5_embedding_range.end, txt_seq_len:] = (
                    image_mask.view(1, img_seq_len)
                )

                # 3. regional img attends to corresponding txt
                # Note that we reshape to (img_seq_len, 1) to ensure broadcasting works as desired.
                regional_attention_mask[txt_seq_len:, t5_embedding_range.start : t5_embedding_range.end] = (
                    image_mask.view(img_seq_len, 1)
                )

                # 4. regional img attends to itself
                image_mask = image_mask.view(img_seq_len, 1)
                regional_attention_mask[txt_seq_len:, txt_seq_len:] += image_mask @ image_mask.T
            else:
                # We don't allow attention between non-background image regions and global prompts. This helps to ensure
                # that regions focus on their local prompts. We do, however, allow attention between background regions
                # and global prompts. If we didn't do this, then the background regions would not attend to any txt
                # embeddings, which we found experimentally to cause artifacts.

                # 2. global txt attends to background region
                # Note that we reshape to (1, img_seq_len) to ensure broadcasting works as desired.
                regional_attention_mask[t5_embedding_range.start : t5_embedding_range.end, txt_seq_len:] = (
                    background_region_mask.view(1, img_seq_len)
                )

                # 3. background region attends to global txt
                # Note that we reshape to (img_seq_len, 1) to ensure broadcasting works as desired.
                regional_attention_mask[txt_seq_len:, t5_embedding_range.start : t5_embedding_range.end] = (
                    background_region_mask.view(img_seq_len, 1)
                )

        # Allow background regions to attend to themselves.
        regional_attention_mask[txt_seq_len:, txt_seq_len:] += background_region_mask.view(img_seq_len, 1)
        regional_attention_mask[txt_seq_len:, txt_seq_len:] += background_region_mask.view(1, img_seq_len)

        # Convert attention mask to boolean.
        regional_attention_mask = regional_attention_mask > 0.5

        return regional_attention_mask

    @classmethod
    def _concat_regional_text_conditioning(
        cls,
        text_conditionings: list[FluxTextConditioning],
        redux_conditionings: list[FluxReduxConditioning],
    ) -> FluxRegionalTextConditioning:
        """Concatenate regional text conditioning data into a single conditioning tensor (with associated masks)."""
        concat_text_embeddings: list[torch.Tensor] = []
        concat_text_embedding_ranges: list[Range] = []
        concat_txt_ids: list[torch.Tensor] = []
        image_masks: list[torch.Tensor | None] = []

        uses_qwen3 = any(text_conditioning.qwen3_embeddings is not None for text_conditioning in text_conditionings)
        if uses_qwen3:
            for text_conditioning in text_conditionings:
                if text_conditioning.qwen3_embeddings is None:
                    raise ValueError("Mixed FLUX text conditioning types detected. Expected Qwen3 embeddings.")
        else:
            for text_conditioning in text_conditionings:
                if text_conditioning.t5_embeddings is None:
                    raise ValueError("Missing T5 embeddings for FLUX text conditioning.")

        # Choose global CLIP embedding.
        # Use the first global prompt's CLIP embedding as the global CLIP embedding. If there is no global prompt, use
        # the first prompt's CLIP embedding.
        global_clip_embedding: torch.Tensor = text_conditionings[0].clip_embeddings
        for text_conditioning in text_conditionings:
            if text_conditioning.mask is None:
                global_clip_embedding = text_conditioning.clip_embeddings
                break

        # Handle text embeddings.
        cur_text_embedding_len = 0
        for text_conditioning in text_conditionings:
            embeddings = (
                text_conditioning.qwen3_embeddings if uses_qwen3 else text_conditioning.t5_embeddings
            )
            assert embeddings is not None
            concat_text_embeddings.append(embeddings)
            concat_text_embedding_ranges.append(
                Range(start=cur_text_embedding_len, end=cur_text_embedding_len + embeddings.shape[1])
            )
            image_masks.append(text_conditioning.mask)
            if uses_qwen3:
                txt_ids = text_conditioning.qwen3_txt_ids
                if txt_ids is None:
                    txt_ids = cls._build_qwen3_txt_ids(
                        embeddings.shape[1], cur_text_embedding_len, embeddings.device, embeddings.dtype
                    )
                else:
                    txt_ids = txt_ids.clone()
                    txt_ids[..., 0] += cur_text_embedding_len
                concat_txt_ids.append(txt_ids)
            cur_text_embedding_len += embeddings.shape[1]

        # Handle Redux embeddings.
        for redux_conditioning in redux_conditionings:
            concat_text_embeddings.append(redux_conditioning.redux_embeddings)
            concat_text_embedding_ranges.append(
                Range(
                    start=cur_text_embedding_len,
                    end=cur_text_embedding_len + redux_conditioning.redux_embeddings.shape[1],
                )
            )
            image_masks.append(redux_conditioning.mask)
            if uses_qwen3:
                concat_txt_ids.append(
                    cls._build_qwen3_txt_ids(
                        redux_conditioning.redux_embeddings.shape[1],
                        cur_text_embedding_len,
                        redux_conditioning.redux_embeddings.device,
                        redux_conditioning.redux_embeddings.dtype,
                    )
                )
            cur_text_embedding_len += redux_conditioning.redux_embeddings.shape[1]

        text_embeddings = torch.cat(concat_text_embeddings, dim=1)

        # Initialize the txt_ids tensor.
        if uses_qwen3:
            txt_ids = torch.cat(concat_txt_ids, dim=1)
            t5_txt_ids = None
            qwen3_txt_ids = txt_ids
        else:
            pos_bs, pos_t5_seq_len, _ = text_embeddings.shape
            t5_txt_ids = torch.zeros(
                pos_bs, pos_t5_seq_len, 3, dtype=text_embeddings.dtype, device=TorchDevice.choose_torch_device()
            )
            qwen3_txt_ids = None

        return FluxRegionalTextConditioning(
            t5_embeddings=text_embeddings if not uses_qwen3 else None,
            qwen3_embeddings=text_embeddings if uses_qwen3 else None,
            clip_embeddings=global_clip_embedding,
            t5_txt_ids=t5_txt_ids,
            qwen3_txt_ids=qwen3_txt_ids,
            image_masks=image_masks,
            t5_embedding_ranges=concat_text_embedding_ranges,
        )

    @staticmethod
    def _build_qwen3_txt_ids(
        seq_len: int, offset: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        txt_ids = torch.zeros((1, seq_len, 3), device=device, dtype=dtype)
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        txt_ids[..., 0] = positions + offset
        return txt_ids

    @staticmethod
    def preprocess_regional_prompt_mask(
        mask: Optional[torch.Tensor], packed_height: int, packed_width: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Preprocess a regional prompt mask to match the target height and width.
        If mask is None, returns a mask of all ones with the target height and width.
        If mask is not None, resizes the mask to the target height and width using 'nearest' interpolation.

        packed_height and packed_width are the target height and width of the mask in the 'packed' latent space.

        Returns:
            torch.Tensor: The processed mask. shape: (1, 1, packed_height * packed_width).
        """

        if mask is None:
            return torch.ones((1, 1, packed_height * packed_width), dtype=dtype, device=device)

        mask = to_standard_float_mask(mask, out_dtype=dtype)

        tf = torchvision.transforms.Resize(
            (packed_height, packed_width), interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )

        # Add a batch dimension to the mask, because torchvision expects shape (batch, channels, h, w).
        mask = mask.unsqueeze(0)  # Shape: (1, h, w) -> (1, 1, h, w)
        resized_mask = tf(mask)

        # Flatten the height and width dimensions into a single image_seq_len dimension.
        return resized_mask.flatten(start_dim=2)
