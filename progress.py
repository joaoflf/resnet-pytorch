from tqdm import tqdm
import time

processing_message = "EPOCH {} - loss: {:.2f}"
for i in range(25):

    pbar = tqdm(
        total=10,
        desc=processing_message.format(0, 9.2),
        position=i
    )
    for j in range(10):
        time.sleep(0.1)
        pbar.update(1)
    pbar.close()

for i in range(int(24/2)):
    print('\n')

print('\n Test Message')
