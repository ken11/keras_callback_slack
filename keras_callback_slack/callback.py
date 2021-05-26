import io
from tensorflow.keras.callbacks import Callback
from slack_sdk.webhook import WebhookClient
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import matplotlib.pyplot as plt


class SlackNotifications(Callback):

    def __init__(self,
                 url,
                 token=None,
                 channel=None,
                 loss_metrics=['loss', 'val_loss'],
                 acc_metrics=[],
                 attachment_image=False,
                 period=1):
        self.url = url
        self.token = token
        self.channel = channel
        self.loss_metrics = loss_metrics
        self.acc_metrics = acc_metrics
        self.attachment_image = attachment_image
        self.period = period
        self.history = dict(zip(
            loss_metrics + acc_metrics, [[]] * len(loss_metrics + acc_metrics)
        ))

    def on_train_begin(self, logs={}):
        with io.StringIO() as buf:
            self.model.summary(print_fn=lambda x: buf.write(x + "\n"))
            text = buf.getvalue()
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Start training the model"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"```{text}```"
                }
            }
        ]
        self._send(blocks)

    def on_epoch_end(self, epoch, logs={}):
        # update history
        for loss_metric in self.loss_metrics:
            self.history[loss_metric] = \
                self.history[loss_metric] + [logs.get(loss_metric)]
        for acc_metric in self.acc_metrics:
            self.history[acc_metric] = \
                self.history[acc_metric] + [logs.get(acc_metric)]

        if (epoch + 1) % self.period == 0:
            fields = self._make_fields(logs)
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"Epoch: {epoch + 1}"
                    }
                },
                {
                    "type": "section",
                    "fields": fields
                }
            ]
            self._send(blocks)
            if self.attachment_image:
                self._make_graph(epoch)

    def on_train_end(self, logs={}):
        fields = self._make_fields(logs)
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Training completed"
                }
            },
            {
                "type": "section",
                "fields": fields
            }
        ]
        self._send(blocks)

    def _make_fields(self, logs):
        fields = []
        for loss_metric in self.loss_metrics:
            fields.append({
                "type": "mrkdwn",
                "text": f"*{loss_metric}:*\n{logs.get(loss_metric):.4f}"
            })
        for acc_metric in self.acc_metrics:
            fields.append({
                "type": "mrkdwn",
                "text": f"*{acc_metric}:*\n{logs.get(acc_metric):.4f}"
            })
        return fields

    def _make_graph(self, epoch):
        try:
            assert self.token is not None
            assert self.channel is not None
        except AssertionError:
            print(
                'SlackNotificationsError: The token and channel options \
                are required when using the attachment_image option.')
            return

        client = WebClient(token=self.token)
        metrics = {"loss": self.loss_metrics, "accuracy": self.acc_metrics}
        plt.figure(figsize=(21, 9))

        for i, (name, metrics) in enumerate(metrics.items()):
            plt.subplot(1, 2, i + 1)
            plt.title(name)
            for metric in metrics:
                plt.plot(
                    range(1, epoch + 2),
                    self.history[metric],
                    label=metric, marker="o"
                )
            plt.legend()

        image_io = io.BytesIO()
        plt.savefig(image_io, format='png')
        plt.close()

        try:
            response = client.files_upload(
                channels=self.channel, file=image_io.getvalue())
            assert 'file' in response
            assert response["file"]
        except SlackApiError as e:
            assert e.response["ok"] is False
            assert e.response["error"]
            print(
                f"SlackNotificationsError: Post image failed.\n{e.response['error']}")
        except AssertionError:
            print("SlackNotificationsError: Post image failed.")

    def _send(self, blocks):
        webhook = WebhookClient(self.url)
        try:
            response = webhook.send(
                text="keras callback notification", blocks=blocks)
            assert response.status_code == 200
            assert response.body == "ok"
        except AssertionError:
            print(f"SlackNotificationsError: Post message failed.\n{response.body}")
