"""Tests for Model Context Protocol (MCP) InsightExtractor Server."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.insight_extractor import InsightExtractor


@pytest.fixture
def mcp_server():
    """Instantiate a clean MCP server component instance for every test block."""
    return InsightExtractor()


class TestMCPServerLifecycle:
    def test_initial_manifest_registration(self, mcp_server):
        """Verify the server initializes with correct JSON-RPC tool schemas."""
        assert mcp_server.server_status == "INITIALIZING"
        assert len(mcp_server.exposed_tools) == 1
        assert mcp_server.exposed_tools[0]["name"] == "extract_enterprise_insights"

    def test_successful_handshake(self, mcp_server):
        """Ensure valid context tokens spin up the protocol interface smoothly."""
        response = mcp_server.initialize_mcp_server(scoped_token="env_session_token")
        assert response["status"] == "RUNNING"
        assert response["protocol_version"] == "2024.11.0"
        assert len(response["manifest"]) == 1

    def test_rejected_handshake_raises_error(self, mcp_server):
        """Verify server switches to error states on invalid session keys."""
        with pytest.raises(ConnectionRefusedError, match="Invalid handshake credential"):
            mcp_server.initialize_mcp_server(scoped_token="compromised_or_malformed_key")
        assert mcp_server.server_status == "SHUTDOWN"


class TestMCPAgentSecurityGating:
    def test_tool_execution_with_authorized_token(self, mcp_server):
        """Verify payload returns correctly under secure execution parameters."""
        mcp_server.initialize_mcp_server(scoped_token="env_session_token")
        
        arguments = {"text_payload": "Revenue grew 15% to $2.3 million."}
        response = mcp_server.execute_tool(
            tool_name="extract_enterprise_insights",
            arguments=arguments,
            token="env_session_token"
        )
        
        assert "tool_output" in response
        assert response["tool_output"]["data_integrity_status"] == "VERIFIED"

    def test_tool_execution_denied_on_unauthorized_token(self, mcp_server):
        """Ensure decorators flag security violations immediately across boundaries."""
        mcp_server.initialize_mcp_server(scoped_token="env_session_token")
        
        arguments = {"text_payload": "Intercepted corporate asset strings."}
        with pytest.raises(PermissionError, match="Unauthorized token context"):
            mcp_server.execute_tool(
                tool_name="extract_enterprise_insights",
                arguments=arguments,
                token="invalid_malicious_token"
            )


class TestMCPPayloadExtractionLogic:
    def test_structured_response_parsing(self, mcp_server):
        """Verify the output tool payload structures are properly formatted."""
        sample_doc = "Revenue grew 15% to $2.3 million. Management must develop expansion plans."
        
        response = mcp_server.execute_tool(
            tool_name="extract_enterprise_insights",
            arguments={"text_payload": sample_doc},
            token="env_session_token"
        )
        
        output = response["tool_output"]
        assert "metrics" in output
        assert "sentiment" in output
        assert "action_items" in output
        
        # Verify specific structural extractions are tracked correctly
        assert output["metrics"]["percentages"][0]["value"] == 15.0
        assert output["sentiment"]["label"] == "positive"
        assert len(output["action_items"]) == 1

    def test_invalid_tool_name_routing(self, mcp_server):
        """Ensure requested tools outside manifest schemas trip routing alerts."""
        with pytest.raises(ValueError, match="not defined in manifest"):
            mcp_server.execute_tool(
                tool_name="nonexistent_malicious_scraping_utility",
                arguments={"data": "test"},
                token="env_session_token"
            )
