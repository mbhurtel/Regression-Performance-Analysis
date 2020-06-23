import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

mapping = {0:"a", 1:"b", 2:"c"}

#plots Definnition
# 3 subplots individual for a,b,c
def exp_vs_pred_subplots(y_test, y_pred, title):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
    for i in range(3):
        ax[i].scatter(y_test[:,i], y_pred[:,i], c='w', edgecolors='b')
        ax[i].plot(y_test[:,i],y_test[:,i], c='r')
        ax[i].set_xlabel(f"Experimental {mapping[i]}")
        ax[i].set_ylabel(f"Predicted {mapping[i]}")

    fig.suptitle(title)
    plt.show()

#1 plot for Lattice Constant only
def exp_vs_pred_lc(y_test, y_pred, title, result):
    plt.scatter(y_test[:,0], y_pred[:,0], c='w', edgecolors='b')
    plt.plot(y_test[:,0],y_test[:,0], c='r')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(min(y_test[:,0]), max(y_pred[:,0]), f"R2 Score: {result}", fontsize=12,
            verticalalignment='top', bbox=props)
    plt.xlabel(f"Experimental Lattice Constant")
    plt.ylabel(f"Predicted Lattice Constant")
    plt.title(title)
    plt.show()

#with hover data in plotly    
def exp_vs_pred_plotly(y_test, y_pred, regressor, names=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test[:,0], y=y_pred[:,0],
                        mode='markers', text=names))
    fig.add_trace(go.Scatter(x=y_test[:,0], y=y_test[:,0],
                        mode='lines'))
    fig.update_layout(title_text = regressor)
    fig.show()

def comparision(r2_scores):
    x_plot = list(r2_scores.keys())
    y_plot= list(r2_scores.values())
    plt.figure(figsize=(18,8))
    plt.title("R2 Scores")
    plt.xlabel("Algorithm")
    plt.ylabel("R2 Score")
    g=sns.barplot(x_plot, y_plot)
    for i,p in enumerate(g.patches):
            percentage = '{:.3f}'.format(y_plot[i])
            x1 = p.get_x() + p.get_width() - 0.5
            y1 = p.get_y() + p.get_height() + 0.02
            g.annotate(percentage, (x1, y1))
    
    plt.show()