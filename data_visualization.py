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

    def __init__(self,num_plots=2):
        print("Something")
        self.f,self.sub_plt = plt.subplots(ncols=num_plots)
        self.i = -1
        self.j= -1
        self.f, self.axarr = plt.subplots(3, 4)
        plt.axis('tight')
        #plt.subplots_adjust(left=0.8, right=0.9, top=0.9, bottom=0.8)
        self.f.tight_layout()  # Or equivalently,  "plt.tight_layout()"
        # self.scale_colors()

    def scale_colors(self):
        import struct
        # Scaling RGB values to [0.1] for matplotlib
        for i in range(len(self.color_pallet)):
            r, g, b = self.color_pallet[i]
            self.color_pallet[i] = (r / 255., g / 255., b / 255.)
            # self.color_pallet[i] = "#"+struct.pack('BBB',*rgb).encode('hex')

    def set_label(self, x_label,y_label):
        rect = 10, 10, 10,10
        self.f.add_axes(rect, label='axes1')
        self.f.add_axes(rect, label='axes2')

    def scatter_plot(self, x, y,df,fit_reg=False):
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
        #sns.set(color_codes=True)
        #self.ax = sns.regplot(x=x, y=y, data=df)
        #sns.regplot(x=x, y=y,
        #           data=df,
        #           fit_reg=fit_reg)
        #self.i = self.i+1
        self.axarr[self.i,(self.j)%4].scatter(df[x], df[y],s=2)
        self.axarr[self.i, (self.j) % 4].set_title("Circuit "+str(self.j+1), size=8)
        self.axarr[self.i, (self.j) % 4].tick_params(labelsize=6)
        #ax.set_ylabel('Year', fontsize=20.0)  # Y label
        #ax.set_xlabel('Active Cdc2-cyclin B', fontsize=20)  # X label
        #self.axarr[self.i, (self.j) % 4].xlabel('xlabel', fontsize=9)
        #self.axarr[self.i, (self.j) % 4].ylabel('ylabel', fontsize=9)
        if self.i ==2 and self.j%4 == 3:
            print(df)
        if(fit_reg):
            self.axarr[self.i, (self.j)%4].plot(df[x], df[y],linewidth=0.7)
            self.axarr[self.i, (self.j) % 4]
        #sns.lmplot(x="year", y="Circuit", hue="e_x_dem", data=df)

    def pointplot(self, x, y,df,):
        self.ax = sns.pointplot(x=x, y=y, data=df)

    def increment(self):
        self.j = self.j + 1
        self.i = int((self.j) / 4)
        print("j:" + str(self.j))
        print("i:" + str(self.i) + " j:" + str(self.j % 4))

    def set_title(self,title):
        plt.suptitle(title, fontsize=12)

    def set_y_axis_label(self,label,axis,left=True):
        plt.set_ylabel(label)
        if axis == "x" and left:
            plt.yaxis.set_label_position("left")
        if axis == "x" and not left:
            plt.yaxis.set_label_position("right")

    def show_plot(self):
        plt.show()

    def bar_chart(self, save=False):
        import random
        data = random.sample(range(1, 100000), 100)
        data2 = random.sample(range(1, 200000), 100)
        ax, plt = self.set_plot_properties()

        from random import randint
        plt.hist(list(data) + list(data2),
                 color=self.color_pallet[randint(0, len(self.color_pallet) - 1)], bins=100)

        plt.show()

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