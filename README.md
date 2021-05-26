# keras_callback_slack
keras custom callback for slack

## Installation
```sh
$ pip install keras-callback-slack
```

## Usage
```py
from keras_callback_slack import SlackNotifications

slack_notification = SlackNotifications(
    'WEBHOOK_URL', token='TOKEN', channel='CHANNEL', attachment_image=True)

model.fit(x, y, batch_size=64, epochs=5, callbacks=[slack_notification])
```

## Arguments
- `url` : string - Slack Incoming Webhook URL
- `token` : string(Optional) - Slack bot token
  This will be needed when uploading the graph.
- `channel` : string(Optional) - Slack channel ID
  The ID of the channel to which you want to upload the graph. Specify the same channel as the webhook.
- `loss_metrics` : list(Optional) - Loss metrics you want to monitor.
  ex. `['loss', 'val_loss']`
- `acc_metrics` : list(Optional) - Acc metrics you want to monitor.
  ex. `['acc', 'val_acc']`
- `attachment_image` : bool(Optional) - Whether to upload the graph.
- `period` : int(Optional) - Notification interval (epochs).

## Example
![Screenshot from 2021-05-27 00-47-49](https://user-images.githubusercontent.com/2043460/119692068-0f81ec00-be86-11eb-92f5-25c824a59414.png)

----

![Screenshot from 2021-05-27 00-48-31](https://user-images.githubusercontent.com/2043460/119692070-10b31900-be86-11eb-9e46-f808a51ceb0d.png)
