import os
import sys
import unittest
from unittest import mock
from test.support import captured_stdout
from slack_sdk.webhook import WebhookResponse
from slack_sdk.errors import SlackApiError

sys.path.append(os.path.abspath(".."))
import keras_callback_slack


class TestCallback(unittest.TestCase):

    def _mock_send(self, text, blocks):
        res = WebhookResponse(url='', status_code=200, body='ok', headers={})
        return res

    def _mock_send_fail(self, text, blocks):
        res = WebhookResponse(
            url='', status_code=401, body='invalid_token', headers={})
        return res

    def _mock_files_upload(self, channels, file):
        res = {"ok": True, "file": {"id": "F0TD00400"}}
        return res

    def _mock_files_upload_fail(self, channels, file):
        res = {"ok": False, "error": "invalid_auth"}
        return res

    def _mock_files_upload_api_error(self, channels, file):
        raise SlackApiError('hoge', {"ok": False, "error": "invalid_auth"})

    @mock.patch(
        "keras_callback_slack.callback.WebhookClient.send", new=_mock_send)
    def test_on_train_begin(self):
        callback = keras_callback_slack.SlackNotifications(
            'https://example.com')
        callback.model = mock.MagicMock()
        self.assertIsNone(callback.on_train_begin())

    @mock.patch(
        "keras_callback_slack.callback.WebhookClient.send",
        new=_mock_send_fail)
    def test_on_train_begin_fail(self):
        callback = keras_callback_slack.SlackNotifications(
            'https://example.com')
        callback.model = mock.MagicMock()
        with captured_stdout() as stdout:
            callback.on_train_begin()
            lines = stdout.getvalue().splitlines()
        self.assertEqual(
            lines[0], 'SlackNotificationsError: Post message failed.')
        self.assertEqual(lines[1], 'invalid_token')

    @mock.patch(
        "keras_callback_slack.callback.WebhookClient.send", new=_mock_send)
    def test_on_epoch_end(self):
        callback = keras_callback_slack.SlackNotifications(
            'https://example.com')
        self.assertIsNone(callback.on_epoch_end(
            0, logs={'loss': 0.00111, 'val_loss': 0.11111}))

    @mock.patch(
        "keras_callback_slack.callback.WebhookClient.send", new=_mock_send)
    def test_on_train_end(self):
        callback = keras_callback_slack.SlackNotifications(
            'https://example.com')
        self.assertIsNone(callback.on_train_end(
            logs={'loss': 0.00111, 'val_loss': 0.11111}))

    def test__make_fields(self):
        callback = keras_callback_slack.SlackNotifications(
            'https://example.com')
        fields = callback._make_fields({'loss': 0.00111, 'val_loss': 0.11111})
        self.assertEqual(fields,
                         [{"type": "mrkdwn", "text": "*loss:*\n0.0011"},
                          {"type": "mrkdwn", "text": "*val_loss:*\n0.1111"}]
                         )

    @mock.patch(
        "keras_callback_slack.callback.WebClient.files_upload",
        new=_mock_files_upload)
    def test__make_graph(self):
        callback = keras_callback_slack.SlackNotifications(
            'https://example.com', token='hoge', channel='fuga')
        callback.history = {'loss': [0.00111], 'val_loss': [0.11111]}
        self.assertIsNone(callback._make_graph(0))

    @mock.patch(
        "keras_callback_slack.callback.WebClient.files_upload",
        new=_mock_files_upload_fail)
    def test__make_graph_fail(self):
        callback = keras_callback_slack.SlackNotifications(
            'https://example.com', token='hoge', channel='fuga')
        callback.history = {'loss': [0.00111], 'val_loss': [0.11111]}
        with captured_stdout() as stdout:
            callback._make_graph(0)
            lines = stdout.getvalue().splitlines()
        self.assertEqual(
            lines[0], 'SlackNotificationsError: Post image failed.')

    @mock.patch(
        "keras_callback_slack.callback.WebClient.files_upload",
        new=_mock_files_upload_api_error)
    def test__make_graph_api_error(self):
        callback = keras_callback_slack.SlackNotifications(
            'https://example.com', token='hoge', channel='fuga')
        callback.history = {'loss': [0.00111], 'val_loss': [0.11111]}
        with captured_stdout() as stdout:
            callback._make_graph(0)
            lines = stdout.getvalue().splitlines()
        self.assertEqual(
            lines[0], 'SlackNotificationsError: Post image failed.')
        self.assertEqual(lines[1], 'invalid_auth')


if __name__ == '__main__':
    unittest.main(verbosity=2)
