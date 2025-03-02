from core.result import Result


class TestResult:

    def test_result_ok(self):
        """Test creating a successful result."""
        result = Result.ok("Success value")
        assert result.success is True
        assert result.value == "Success value"
        assert result.error is None
        assert result.error_message is None

    def test_result_fail(self):
        """Test creating a failed result with an error."""
        error = ValueError("Test error")
        result = Result.fail(error=error)
        assert result.success is False
        assert result.value is None
        assert result.error == error
        assert result.error_message == "Test error"

    def test_result_fail_with_message(self):
        """Test creating a failed result with a custom message."""
        result = Result.fail(error_message="Custom error message")
        assert result.success is False
        assert result.value is None
        assert result.error is None
        assert result.error_message == "Custom error message"

    def test_result_bool_conversion(self):
        """Test boolean conversion of results."""
        success_result = Result.ok("Value")
        failure_result = Result.fail(error_message="Error")

        assert bool(success_result) is True
        assert bool(failure_result) is False

        # Test in if statement
        if success_result:
            success_flag = True
        else:
            success_flag = False

        if failure_result:
            failure_flag = True
        else:
            failure_flag = False

        assert success_flag is True
        assert failure_flag is False

    def test_result_with_metadata(self):
        """Test results with metadata."""
        metadata = {"key1": "value1", "key2": 123}

        success_result = Result.ok("Value", metadata=metadata)
        assert success_result.metadata == metadata

        failure_result = Result.fail(error_message="Error", metadata=metadata)
        assert failure_result.metadata == metadata
