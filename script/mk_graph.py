import matplotlib.pyplot as plt
import pandas as pd

# Output
output_file_name = "svdj.pdf"
fig_block_m = 1

# The list of `type` in the input csv file
mode_list = ['lapack', 'cusolver']
color_table = {
        'lapack' : 'red',
        'cusolver' : 'orange',
        }

eval_list = ['residual', 'u_orth', 'v_orth']
fig_block_n = len(eval_list)

# Load input data
df = pd.read_csv("data.csv", encoding="UTF-8")

# Create graph
fig, axs = plt.subplots(fig_block_m, fig_block_n, figsize=(10, 3))

line_list = []
label_list = []
for i in range(fig_block_n):
    axs[i].set_xscale('log', base=2)
    axs[i].set_yscale('log', base=10)
    axs[i].set_title(eval_list[i])
    axs[i].set_xlabel('Input matrix size $M$ : $\\mathbf{A} \in \mathrm{float}^{M \\times M}$')
    axs[i].grid(True)
    axs[i].set_ylim(1e-7, 3e-3)
    # Plot
    for t in mode_list:
        df_t = df[df['mode'] == t]
        l = axs[i].plot(
                df_t['m'],
                df_t[eval_list[i]],
                markersize=4,
                marker="*",
                color=color_table[t])
        if i == 0:
            line_list += [l]
            label_list += [t]

        axs[i].annotate('', (2**11, 2e-7), (2**11, 1e-6), arrowprops=dict(facecolor='darkgray', shrink=0., width=2, headwidth=6))
        axs[i].text(1.2 * 2**11 , 5e-7, 'better', ha="center", va="center", rotation=90)

# Legend config
fig.legend(line_list,
           labels=label_list,
           loc='upper center',
           ncol=len(mode_list),
           bbox_to_anchor=(0.5, 1.1)
           )

# Save to file
fig.tight_layout()
fig.subplots_adjust(hspace=0)
fig.savefig(output_file_name, bbox_inches="tight")
