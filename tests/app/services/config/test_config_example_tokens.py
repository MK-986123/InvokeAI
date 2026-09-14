from unittest.mock import patch

import yaml

from invokeai.app.services.config.config_default import get_config
from invokeai.frontend.cli.arg_parser import InvokeAIArgs


def test_example_config_uses_generated_tokens(tmp_path):
    get_config.cache_clear()
    with patch.object(InvokeAIArgs, "did_parse", True), patch.object(InvokeAIArgs, "args") as mock_args:
        mock_args.root = str(tmp_path)
        mock_args.config_file = None

        config = get_config()
        example_file = config.config_file_path.with_suffix(".example.yaml")
        assert example_file.exists()

        with open(example_file, "r") as f:
            data = yaml.safe_load(f)

        assert "remote_api_tokens" in data
        tokens = data["remote_api_tokens"]
        assert len(tokens) == 2
        for pair in tokens:
            token = pair["token"]
            assert token != "my_secret_token"
            assert token != "some_other_token"
            assert len(token) == 32
    get_config.cache_clear()
