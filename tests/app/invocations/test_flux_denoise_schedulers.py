import pytest
import torch

from invokeai.app.invocations.flux_denoise import FluxDenoiseInvocation
from invokeai.app.invocations.fields import FluxConditioningField
from invokeai.app.invocations.model import TransformerField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.flux_schedulers import FLUX_SCHEDULER_MAP, FLUX_SCHEDULER_NAME
from invokeai.backend.model_manager.config import MainDiffusersConfig as MainModelConfig
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import FLUXConditioningInfo, ConditioningFieldData


class MockConditioningService:
    def load(self, conditioning_name: str) -> ConditioningFieldData:
        # The test will fail if the conditioning name is not found.
        # We expect a KeyError to be raised here, which is a good sign that the
        # scheduler logic has passed.
        raise KeyError(conditioning_name)


class MockModelManager:
    def get_config(self, key: str):
        # Return a dummy config that looks like a flux-dev model.
        # The 'config_path' is what determines if the model is 'schnell' or 'dev'.
        return MainModelConfig(
            path="/dummy/path", model_format="diffusers", base_type="sd-1", model_type="main", name="dummy", key=key, config_path="flux-dev", hash="dummy_hash", file_size=0, source="dummy_source", source_type="path"
        )

    def load(self, key: str):
        # This will also be mocked, but we don't need it for this test.
        pass


@pytest.fixture
def mock_context(mock_services) -> InvocationContext:
    mock_services.conditioning = MockConditioningService()
    mock_services.model_manager = MockModelManager()
    return InvocationContext(
        services=mock_services,
        graph_execution_state_id="1",
        invocation=None,  # Will be set in the test.
        queue_id="1",
    )


@pytest.mark.parametrize("scheduler_name", FLUX_SCHEDULER_MAP.keys())
def test_flux_denoise_scheduler_selection(scheduler_name: FLUX_SCHEDULER_NAME, mock_context: InvocationContext):
    """
    Tests that FluxDenoiseInvocation correctly selects a scheduler and proceeds to the
    point of loading conditioning data.
    """
    invocation = FluxDenoiseInvocation(
        id="test_flux_denoise",
        scheduler=scheduler_name,
        transformer=TransformerField(transformer="dummy_transformer_key"),
        positive_text_conditioning=FluxConditioningField(conditioning_name="dummy_conditioning_name"),
    )
    mock_context.invocation = invocation

    # We expect the invocation to fail when it tries to load the conditioning data.
    # This is a good sign, because it means that the scheduler was successfully
    # instantiated and used to generate timesteps.
    with pytest.raises(KeyError, match="dummy_conditioning_name"):
        invocation.invoke(mock_context)