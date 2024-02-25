import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

for s in os.listdir('./results'):
    s_path = os.path.join('./results', s)

    if os.path.isdir(s_path):
        for filename in os.listdir(s_path):
            if 'events.out.tfevents.' in filename:
                event_file = os.path.join(s_path, filename)
                event_acc = EventAccumulator(event_file)
                event_acc.Reload()
                for scalar in event_acc.Tags()['scalars']:
                    try:
                        values = event_acc.Scalars(scalar)
                        if not values:
                            continue
                        steps = [val.step for val in values]
                        vals = [val.value for val in values]
                        plt.figure()
                        plt.plot(steps, vals)
                        plt.title(scalar)
                        img_filename = f'{scalar.replace("/", "_")}.png'
                        img_path = os.path.join(s_path, img_filename)
                        plt.savefig(img_path)
                        plt.close()

                        df = pd.DataFrame({'Step': steps, 'Value': vals})
                        csv_filename = f'{scalar.replace("/", "_")}.csv'
                        csv_path = os.path.join(s_path, csv_filename)
                        df.to_csv(csv_path, index=False)

                    except Exception as e:
                        print(f"Error processing scalar {scalar} in file {event_file}: {e}")
