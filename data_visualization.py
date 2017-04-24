import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns

from matplotlib.ticker import MultipleLocator


class DataVisualization():
    x = []
    y = []
    # Graph Colors
    color_pallet = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    def init(self):
        print("Something")
        # self.scale_colors()

    def scale_colors(self):
        import struct
        # Scaling RGB values to [0.1] for matplotlib
        for i in range(len(self.color_pallet)):
            r, g, b = self.color_pallet[i]
            self.color_pallet[i] = (r / 255., g / 255., b / 255.)
            # self.color_pallet[i] = "#"+struct.pack('BBB',*rgb).encode('hex')

    def set_label(self, x_label,y_label):
        self.ax.set(xlabel=x_label, ylabel=y_label)

    def scatter_plot(self, x, y,df):
        '''
         plt.scatter(x, y)
        self.ax = plt.subplot(111)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.grid(alpha=0.3)
        self.ax.spines['bottom'].set_linewidth(0.5)
        self.ax.spines['left'].set_linewidth(0.5)
        #loc = MultipleLocator(100)  # this locator puts ticks at regular intervals
        #self.ax.xaxis.set_major_locator(loc)
        #self.ax.fill_between(x, 0, y, alpha=0.1)
        #plt.plot(lw=2)
        '''
        sns.set(color_codes=True)
        self.ax = sns.regplot(x=x, y=y, data=df)

    def set_title(self,title):
        plt.suptitle(title, fontsize=15)

    def show_plot(self):
        plt.show()

    def bar_chart(self, save=False):
        import random
        data = random.sample(range(1, 100000), 100)
        data2 = random.sample(range(1, 200000), 100)

        print(self.color_pallet)

        ax, plt = self.set_plot_properties()

        from random import randint
        plt.hist(list(data) + list(data2),
                 color=self.color_pallet[randint(0, len(self.color_pallet) - 1)], bins=100)

        plt.text(1300, -5000, "DSGA3003 Project |\
                              sm7029 | avm358 | pc2310", fontsize=10)
        plt.show()

    def line_curve(self, data, captions, x_column, heading=None, xlimit=None, ylimit=None, ):
        self.scale_colors()
        plt.figure(figsize=(12, 14))

        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        if ylimit is not None:
            plt.ylim(ylimit[0], ylimit[1])
        if xlimit is not None:
            plt.xlim(xlimit[0], xlimit[1])

        # plt.yticks(range(0, 91, 10), [str(x) + "%" for x in range(0, 91, 10)], fontsize=14)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)

        plt.tick_params(axis="both", which="both", bottom="off", top="off",
                        labelbottom="on", left="off", right="off", labelleft="on")

        y_pos = 0.0
        i = 0
        if captions is not None:
            for rank, column in enumerate(captions):
                plt.plot(data[x_column].values,
                         data[column.replace("\n", " ")].values,
                         lw=2.5, color=self.color_pallet[rank])

                y_pos = data[column.replace("\n", " ")].values[-1] - 0.5
                y_pos += 0.8
                plt.text(2011.5, y_pos, column, fontsize=14, color=self.color_pallet[rank])
                i = i + 1

        # matplotlib's title() call centers the title on the plot, but not the graph,
        # Here used text() call to customize where the title goes.
        if heading is not None:
            plt.text(1995, 93, heading, fontsize=17, ha="center")

        plt.text(1966, -8, "\nDSGA3110 Project"
                           "\nsm7029 | avm358 | pc2310", fontsize=10)
        plt.savefig(str(time.time()) + "_line.png", bbox_inches="tight")

    def set_plot_properties(self, ax, plt):
        # plt.spines["top"].set_visible(False)
        # plt.spines["right"].set_visible(False)
        plt.figure(figsize=(8, 6))
        plt.xticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.text(100, -3, "DSGA3003 Project |\
                                      sm7029 | avm358 | pc2310", fontsize=10)
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
        plt.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
        return plt

    def save_plot(self):
        plt.savefig("images/" + str(time.time()) + ".png", bbox_inches="tight");

def __test__():
    a = ['Health Professions', 'Public Administration', 'Education', 'Psychology',
         'Foreign Languages', 'English', 'Communications\nand Journalism',
         'Art and Performance', 'Biology', 'Agriculture',
         'Social Sciences and History', 'Business', 'Math and Statistics',
         'Architecture', 'Physical Sciences', 'Computer Science',
         'Engineering']
    df = pd.read_csv("http://www.randalolson.com/wp-content/uploads/percent-bachelors-degrees-women-usa.csv")
    print(df)
    d = DataVisualization()
    #d.line_curve(data=df, captions=a, x_column="Year")
    y = []
    for i in range(len(df['Health Professions'])):
        y.append(i)
    d.set_title("Test Plot")
    d.scatter_plot("Architecture","Social Sciences and History",df)
    d.scatter_plot("Architecture", "Computer Science", df)
    d.set_label("Archi","CS & SSc")
    d.show_plot()

def main():
    __test__()

if __name__ == "__main__":
    main()