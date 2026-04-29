import unittest

from livepeer_gateway.sse import SSEEvent, parse_sse_lines


class SSEParserTest(unittest.TestCase):
    def test_parses_multiline_data_and_fields(self) -> None:
        event = parse_sse_lines(
            [
                "id: abc",
                "event: token",
                "retry: 5000",
                "data: hello",
                "data: world",
            ]
        )

        self.assertEqual(
            event,
            SSEEvent(
                event="token",
                id="abc",
                retry=5000,
                data="hello\nworld",
            ),
        )

    def test_ignores_comments_and_empty_comment_events(self) -> None:
        self.assertIsNone(parse_sse_lines([": keepalive"]))

    def test_preserves_done_sentinel(self) -> None:
        event = parse_sse_lines(["data: [DONE]"])

        self.assertIsNotNone(event)
        self.assertEqual(event.data, "[DONE]")


if __name__ == "__main__":
    unittest.main()
