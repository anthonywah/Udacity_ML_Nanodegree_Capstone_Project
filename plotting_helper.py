import pandas as pd
import sys
import matplotlib.pyplot as plt

def plot_predictions(predictions_array, actuals_array, chunk_size=2500, title=None):
    plt.style.use('seaborn')
    prev_i = None
    df = pd.DataFrame({'prediction': predictions_array, 'actual': actuals_array}).sort_values('actual').reset_index(drop=True)
    chunks_df = pd.DataFrame()
    data_len = df.shape[0]
    for i in range(0, data_len, chunk_size):
        if prev_i is None:
            prev_i = 0
            continue
        train_preds_chunk = df.loc[range(prev_i, i), 'prediction'].mean()
        train_labels_chunk = df.loc[range(prev_i, i), 'actual'].mean()
        chunks_df = chunks_df.append(pd.DataFrame({'prediction': [train_preds_chunk], 'actual': [train_labels_chunk]})).reset_index(drop=True)
        pct_ind = int((1 + i) * 50 / data_len)
        sys.stdout.write('\r')
        sys.stdout.write('Computing in progress: [{}{}] {}%'.format("=" * pct_ind, "-" * (50 - pct_ind), pct_ind * 100 / 50) + f' | from {prev_i} to {i}')
        prev_i = i

    # Remainders
    prev_i = i
    i = data_len
    train_preds = df.loc[range(prev_i, i), 'prediction'].mean()
    train_labels = df.loc[range(prev_i, i), 'actual'].mean()
    pct_ind = 50
    sys.stdout.write('\r')
    sys.stdout.write('Computing in progress: [{}{}] {}%'.format("=" * pct_ind, "-" * (50 - pct_ind), pct_ind * 100 / 50) + f' | from {prev_i} to {i - 1}')
    chunks_df = chunks_df.append(pd.DataFrame({'prediction': [train_preds], 'actual': [train_labels]})).reset_index(drop=True)
    print('\nFinished computing all predictions')
    
    fig = plt.figure(figsize=(20, 15))
    ax1, ax2 = plt.subplot(211), plt.subplot(212)
    for _ax, _df in [(ax1, df), (ax2, chunks_df)]:
        _ax.scatter(_df['prediction'], _df['actual'])
        _ax.hlines(0, xmin=min(_df['prediction']), xmax=max(_df['prediction']), alpha=0.3)
        _ax.vlines(0, ymin=min(_df['actual']), ymax=max(_df['actual']), alpha=0.3)
        _ax.set_xlabel('prediction')
        _ax.set_ylabel('actual')
        if title:
            _ax.set_title(title)
    plt.show()
    plt.close()
    return

