from src.ai.opencode import OpenCodeClient


def main() -> None:
    client = OpenCodeClient()

    result = client.run(
        """
Return ONLY a JSON object with exactly these fields:
status, message.

Set status to success and message to adapter-test.
""".strip()
    )

    print("=== OpenCode Adapter Test ===")
    print(f"Session: {result.session_id}")
    print(f"Raw text: {result.text}")
    print(f"Parsed data: {result.data}")

    assert isinstance(result.data, dict)
    assert result.data["status"] == "success"
    assert result.data["message"] == "adapter-test"

    print()
    print("OpenCode adapter OK")


if __name__ == "__main__":
    main()