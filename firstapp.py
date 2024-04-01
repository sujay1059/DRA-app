#Import the required Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import numpy as np
from PIL import Image
import time

import streamlit as st

import statsmodels.api as sm
import streamlit as st
from sklearn.linear_model import LinearRegression
import matplotlib.colors as colors
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


st.set_page_config(layout="wide")

# Functions for each of the pages

def interactive_plot():
    col1, col2 = st.columns(2)
    
    x_axis_val = col1.selectbox('Select the X-axis', options=df.columns)
    y_axis_val = col2.selectbox('Select the Y-axis', options=df.columns)

    plot = px.scatter(df, x=x_axis_val, y=y_axis_val)
    st.plotly_chart(plot, use_container_width=True)

def min_max(df):

    columns = len(df.columns)

    required_columns = ['DEPTH']
        #Create the figure
    if all(col in df.columns for col in required_columns):

        cols_to_plot = list(df)
        st.header('Statistics of Dataframe')
        st.write('Description of the Data')
        st.write(df.describe())
        st.write('Headings of the data')
        st.write(df.head())




        rows = round(columns/2)
        cols = 2

        fig,ax=plt.subplots(figsize=(10,10))

        cols_to_plot.remove("DEPTH")
        for i, feature in enumerate(cols_to_plot):
            ax=fig.add_subplot(rows,cols,i+1)
            df[feature].hist(bins=20,ax=ax,facecolor='green', alpha=0.6)
            ax.set_title(feature+" Distribution")
            ax.set_axisbelow(True)
            ax.grid(color='whitesmoke')

        plt.tight_layout()  
        st.pyplot(fig)



def logplots(df):
    
        
        df.replace(-999, np.nan, inplace=True)
        df.dropna(inplace =True)

        fig, ax = plt.subplots(figsize=(15, 10))
        
        
        ax1 = plt.subplot2grid((1, 5), (0, 0), rowspan=1, colspan=1)
        ax2 = plt.subplot2grid((1, 5), (0, 1), rowspan=1, colspan=1, sharey=ax1)
        ax3 = plt.subplot2grid((1, 5), (0, 2), rowspan=1, colspan=1, sharey=ax1)
        ax4 = plt.subplot2grid((1, 5), (0, 3), rowspan=1, colspan=1, sharey=ax1)
        ax5 = ax3.twiny()

        if 'GR' in df.columns:
            ax1.plot("GR", "DEPTH", data = df, color = "green", linewidth = 0.5)
            ax1.set_xlabel("Gamma")
            ax1.xaxis.label.set_color("green")
            ax1.set_ylabel("Depth (m)")
            ax1.tick_params(axis='x', colors="green")
            ax1.spines["top"].set_edgecolor("green")
            ax1.title.set_color('green')
        
        if 'RT' in df.columns:
            ax2.set_xlabel("Resistivity")
            ax2.xaxis.label.set_color("red")
            ax2.tick_params(axis='x', colors="red")
            ax2.spines["top"].set_edgecolor("red")
            ax2.semilogx()
            ax2.plot(df["RT"], df["DEPTH"], color="red", linewidth=0.5)

        if 'RHOB' in df.columns:
            ax3.set_xlabel("Density")
            ax3.xaxis.label.set_color("red")
            ax3.tick_params(axis='x', colors="red")
            ax3.spines["top"].set_edgecolor("red")
            ax3.plot(df["RHOB"], df["DEPTH"], color="red", linewidth=0.5)

        if 'DT' in df.columns:
            ax4.set_xlabel("Sonic")
            ax4.xaxis.label.set_color("purple")
            ax4.tick_params(axis='x', colors="purple")
            ax4.spines["top"].set_edgecolor("purple")
            ax4.plot(df["DT"], df["DEPTH"], color="purple", linewidth=0.5)

        if 'NPHI' in df.columns:
            ax5.set_xlabel('Neutron')
            ax5.xaxis.label.set_color("blue")
            ax5.tick_params(axis='x', colors="blue")
            ax5.spines["top"].set_position(("axes", 1.08))
            ax5.spines["top"].set_visible(True)
            ax5.spines["top"].set_edgecolor("blue")
            ax5.plot(df["NPHI"], df["DEPTH"], color="blue", linewidth=0.5)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.grid(which='major', color='lightgrey', linestyle='-')
            ax.xaxis.set_ticks_position("top")
            ax.xaxis.set_label_position("top")
            ax.spines["top"].set_position(("axes", 1.02))

        for ax in [ax2, ax3, ax4]:
            plt.setp(ax.get_yticklabels(), visible=False)

        plt.tight_layout()
        fig.subplots_adjust(wspace=0.15)
        st.pyplot(fig)

# Example usage:
# logplots(df)
        
    
# Function to plot missing values
def missingvalues(df):
    # Replace NaN with True and existing values with False
    missing_data = df.isnull()

    # Create a heatmap using Seaborn
    plt.figure(figsize=(10, 6))
    sns.heatmap(missing_data, cmap='viridis', cbar=False)
    plt.title('Missing Values in DataFrame')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    
    # Display the plot in Streamlit
    st.pyplot()


def outlier(df):
    
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_columns) == len(df.columns):
    
        red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')

        fig, axs = plt.subplots(1, len(df.columns), figsize=(30,10))

        for i, ax in enumerate(axs.flat):
            ax.boxplot(df.iloc[:,i], flierprops=red_circle)
            ax.set_title(df.columns[i], fontsize=20, fontweight='bold')
            ax.tick_params(axis='y', labelsize=14)
            
            #Checking if column names are equal to columns we expect to be logarithmic
            if df.columns[i] == 'RDEP' or df.columns[i] == 'RMED':
                ax.semilogy()
            
        plt.tight_layout()
        st.pyplot()
    else: 
        st.write("Sorry, unable to generate output. Data format does not match our requirements.")



def normalization(df):
    if 'Well' in df.columns:

        df['WELL'].unique()

        wells = df.groupby('WELL')

        fig, ax = plt.subplots(figsize=(8,6))
        for label, df in wells:
            df.GR.plot(kind ='kde', ax=ax, label=label)
            plt.xlim(0, 200)
        plt.grid(True)
        plt.legend()
        plt.savefig('before_normalisation.png', dpi=300)

        gr_percentile_05 = df.groupby('WELL')['GR'].quantile(0.05)
        print(gr_percentile_05)
        df['05_PERC'] = df['WELL'].map(gr_percentile_05)
        gr_percentile_95 = df.groupby('WELL')['GR'].quantile(0.95)
        df['95_PERC'] = df['WELL'].map(gr_percentile_95)


        def normalise(curve, ref_low, ref_high, well_low, well_high):
            return ref_low + ((ref_high - ref_low) * ((curve - well_low) / (well_high - well_low)))

        key_well_low = 25.6464
        key_well_high = 110.5413

        df['GR_NORM'] = df.apply(lambda x: normalise(x['GR'], key_well_low, key_well_high, 
                                                        x['05_PERC'], x['95_PERC']), axis=1)
            
        fig, ax = plt.subplots(figsize=(8,6))
        for label, df in wells:
            df.GR_NORM.plot(kind ='kde', ax=ax, label=label)
            plt.xlim(0, 200)
        plt.grid(True)
        plt.legend()
        plt.savefig('after_normalisation.png', dpi=300)
    else: 
        st.write("Sorry, unable to generate output. Data format does not match our requirements.")


def poro(df):
    required_columns = ['CPOR','DEPTH','CKHG']
        #Create the figure
    if all(col in df.columns for col in required_columns):

    

        x = df['CPOR']
        x = sm.add_constant(x)
        y   = np.log10(df['CKHG'])
        st.header('Plot of Data')

        model = sm.OLS(y, x, missing='drop')
        results = model.fit()
        
        fig, ax = plt.subplots(1,1)
        from matplotlib.ticker import FuncFormatter

        ax.axis([0, 30, 0.01, 100000])
        ax.semilogy(df['CPOR'], df['CKHG'], 'bo')

        ax.grid(True)
        ax.set_ylabel('Core Perm (mD)')
        ax.set_xlabel('Core Porosity (%)')

        ax.semilogy(df['CPOR'], 10**(results.params[1] * df['CPOR'] + results.params[0]), 'r-')

    #Format the axes so that they show whole numbers
        for axis in [ax.yaxis, ax.xaxis]:
            formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
            axis.set_major_formatter(formatter)    
        
        st.pyplot(fig)
    else: 
        st.write("Sorry, unable to generate output. Data format does not match our requirements.")
    

def poroperm_detailed(df):
    required_columns = ['CPOR','DEPTH','CKHG','CGD']
        #Create the figure
    if all(col in df.columns for col in required_columns):
        fig, ax = plt.subplots(figsize=(10,10))

        #Add the axes / subplots using subplot2grid
        ax1 = plt.subplot2grid(shape=(3,3), loc=(0,0), rowspan=3)
        ax2 = plt.subplot2grid(shape=(3,3), loc=(0,1), rowspan=3)
        ax3 = plt.subplot2grid(shape=(3,3), loc=(0,2))
        ax4 = plt.subplot2grid(shape=(3,3), loc=(1,2))
        ax5 = plt.subplot2grid(shape=(3,3), loc=(2,2))

        #Add ax1 to show CPOR (Core Porosity) vs DEPTH
        ax1.scatter(df['CPOR'], df['DEPTH'], marker='.', c='red')
        ax1.set_xlim(0, 50)
        ax1.set_ylim(4010, 3825)
        ax1.set_title('Core Porosity')
        ax1.grid()

        #Add ax2 to show CKHG (Core Permeability) vs DEPTH
        ax2.scatter(df['CKHG'], df['DEPTH'], marker='.', c='blue')
        ax2.set_xlim(0.01, 10000)
        ax2.semilogx()
        ax2.set_ylim(4010, 3825)
        ax2.set_title('Core Permeability')
        ax2.grid()

        #Add ax3 to show CPOR (Core Porosity) vs CKHG (Core Permeability)
        ax3.scatter(df['CPOR'], df['CKHG'], marker='.', alpha=0.5)
        ax3.semilogy()
        ax3.set_xlim(0.01, 10000)
        ax3.set_xlim(0,50)
        ax3.set_title('Poro-Perm Scatter Plot')
        ax3.set_xlabel('Core Porosity (%)')
        ax3.set_ylabel('Core Permeability (mD)')
        ax3.grid()

        #Add ax4 to show a histogram of CPOR - Core Porosity
        ax4.hist(df['CPOR'], bins=30, edgecolor='black', color='red', alpha=0.6)
        ax4.set_xlabel('Core Porosity')

        #Add ax5 to show a histogram of CGD - Core Grain Density
        ax5.hist(df['CGD'], bins=30, edgecolor='black', color='blue', alpha=0.6)
        ax5.set_xlabel('Core Grain Density')

        plt.tight_layout()
        st.pyplot(fig)
    else: 
        st.write("Sorry, unable to generate output. Data format does not match our requirements.")


def semilog(df):
    required_columns = ['CPOR','DEPTH','CKHL','CGD']
        #Create the figure
    if all(col in df.columns for col in required_columns):   
        df = df[['DEPTH', 'CPOR', 'CKHL', 'CGD']]
        df.dropna(inplace=True)

        x = df['CPOR'].values
        y = np.log10(df['CKHL'].values)

        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        if len(x) == 0:
            st.write("No valid data for regression.")
            return
        model = LinearRegression()
        model.fit(x, y)
        r2 = model.score(x, y)
        x_plot_vals = np.arange(0, 50)
        y_pred = model.predict(x_plot_vals.reshape(-1,1))
        y_pred_log = 10**y_pred
        results_df = pd.DataFrame({'por_vals': x_plot_vals, 'perm_vals': y_pred_log.flatten()})

        fig,ax = plt.subplots()
        p = ax.scatter(x=df['CPOR'], y=df['CKHL'], c=df['CGD'], cmap='YlOrRd', s=50)
        plt.colorbar(p,ax=ax)
        plt.plot(results_df['por_vals'], results_df['perm_vals'], color='black')
        plt.semilogy()
        plt.ylabel('Core Permeability (mD)', fontsize=12, fontweight='bold')
        ax.grid(True)
        plt.xlabel('Core Porosity (%)', fontsize=12, fontweight='bold');

        st.write("Semilog Plot:")
        st.pyplot(fig)
    else: 
        st.write("Sorry, unable to generate output. Data format does not match our requirements.")

def boxplots(df):
    required_columns = ['LITH']
        #Create the figure
    if all(col in df.columns for col in required_columns):
        flierprops = dict(marker='o', markersize=5, markeredgecolor='black', markerfacecolor='green', alpha=0.5)
        p = sns.boxplot(y=df['LITH'], x=df['GR'], flierprops=flierprops)
        p.set_xlabel('Gamma Ray', fontsize= 14, fontweight='bold')
        p.set_ylabel('Lithology', fontsize= 14, fontweight='bold')
        p.set_title('Gamma Ray Distribution by Lithology', fontsize= 16, fontweight='bold')
        st.pyplot()
    else: 
        st.write("Sorry, unable to generate output. Data format does not match our requirements.")


def Unsupervised_Clustering_for_Lithofacies(df):

    required_columns = ["WELL", "DEPTH_MD", "RDEP", "RHOB", "GR", "NPHI", "PEF", "DTC", "FORCE_2020_LITHOFACIES_LITHOLOGY"]
        #Create the figure
    if all(col in df.columns for col in required_columns):

        workingdf = df[["WELL", "DEPTH_MD", "RDEP", "RHOB", "GR", "NPHI", "PEF", "DTC", "FORCE_2020_LITHOFACIES_LITHOLOGY"]].copy()
        workingdf.rename(columns={'FORCE_2020_LITHOFACIES_LITHOLOGY':'FACIES'}, inplace=True)

        lithology_numbers = {30000: 'Sandstone',
                        65030: 'Sandstone/Shale',
                        65000: 'Shale',
                        80000: 'Marl',
                        74000: 'Dolomite',
                        70000: 'Limestone',
                        70032: 'Chalk',
                        88000: 'Halite',
                        86000: 'Anhydrite',
                        99000: 'Tuff',
                        90000: 'Coal',
                        93000: 'Basement'}

        simple_lithology_numbers = {30000: 1,
                        65030: 2,
                        65000: 3,
                        80000: 4,
                        74000: 5,
                        70000: 6,
                        70032: 7,
                        88000: 8,
                        86000: 9,
                        99000: 10,
                        90000: 11,
                        93000: 12}
        workingdf['LITH'] = workingdf['FACIES'].map(lithology_numbers)
        workingdf['LITH_SI'] = workingdf['FACIES'].map(simple_lithology_numbers)

        g = sns.FacetGrid(workingdf, col='LITH', col_wrap=4)
        g.map(sns.scatterplot, 'NPHI', 'RHOB', alpha=0.5)
        g.set(xlim=(-0.15, 1))
        g.set(ylim=(3, 1))


        def create_plot(wellname, dataframe, curves_to_plot, depth_curve, log_curves=[], facies_curves=[]):
        # Count the number of tracks we need
            num_tracks = len(curves_to_plot)
            
            facies_color = ['#F4D03F', '#F5B041','#DC7633','#6E2C00', '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D', 'red','black', 'blue']
            
                    
            # Setup the figure and axes
            fig, ax = plt.subplots(nrows=1, ncols=num_tracks, figsize=(num_tracks*2, 10))
            
            # Create a super title for the entire plot
            fig.suptitle(wellname, fontsize=20, y=1.05)
            
            # Loop through each curve in curves_to_plot and create a track with that data
            for i, curve in enumerate(curves_to_plot):
                if curve in facies_curves:
                    cmap_facies = colors.ListedColormap(facies_color[0:dataframe[curve].max()], 'indexed')
                    
                    cluster=np.repeat(np.expand_dims(dataframe[curve].values,1), 100, 1)
                    im=ax[i].imshow(cluster, interpolation='none', cmap=cmap_facies, aspect='auto',vmin=dataframe[curve].min(),vmax=dataframe[curve].max(), 
                                    extent=[0,20, depth_curve.max(), depth_curve.min()])
                    
        #             for key in lithology_setup.keys():
        #                 color = lithology_setup[key]['color']
        #                 ax[i].fill_betweenx(depth_curve, 0, dataframe[curve].max(), 
        #                                   where=(dataframe[curve]==key),
        #                                   facecolor=color)
        #                 
                else:
                    ax[i].plot(dataframe[curve], depth_curve)

                
                # Setup a few plot cosmetics
                ax[i].set_title(curve, fontsize=14, fontweight='bold')
                ax[i].grid(which='major', color='lightgrey', linestyle='-')
                
                # We want to pass in the deepest depth first, so we are displaying the data 
                # from shallow to deep
                ax[i].set_ylim(depth_curve.max(), depth_curve.min())
        #         ax[i].set_ylim(3500, 3000)

                # Only set the y-label for the first track. Hide it for the rest
                if i == 0:
                    ax[i].set_ylabel('DEPTH (m)', fontsize=18, fontweight='bold')
                else:
                    plt.setp(ax[i].get_yticklabels(), visible = False)
                
                # Check to see if we have any logarithmic scaled curves
                if curve in log_curves:
                    ax[i].set_xscale('log')
                    ax[i].grid(which='minor', color='lightgrey', linestyle='-')
                

            
            plt.tight_layout()
            plt.show()
            
            return cmap_facies




        def well_splitter(dataframe, groupby_column):
            grouped = dataframe.groupby(groupby_column)
            
            # Create empty lists
            wells_as_dfs = []
            wells_wellnames = []

            #Split up the data by well
            for well, data in grouped:
                wells_as_dfs.append(data)
                wells_wellnames.append(well)

            print('index  wellname')
            for i, name in enumerate(wells_wellnames):
                print(f'{i}      {name}')
            
            return wells_as_dfs, wells_wellnames
        grouped_wells, grouped_names = well_splitter(workingdf, 'WELL')
        def optimise_k_means(data, max_k):
            means = []
            inertias = []
            
            for k in range(1,max_k):
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(data)
                means.append(k)
                inertias.append(kmeans.inertia_)
                
            fig = plt.subplots(figsize=(10, 5))
            plt.plot(means, inertias, 'o-')
            plt.xlabel("Number of Clusters")
            plt.ylabel("Inertia")
            plt.grid(True)
            plt.show()

        workingdf.dropna(inplace =True)

        from yellowbrick.cluster import SilhouetteVisualizer

        def visualise_k_means_sillouette(data, max_k):
            fig, ax = plt.subplots(2, 3, figsize=(10, 5))
            means = []
            silhouette_avg = []
            
            for k in range(2,max_k): #start at 2 clusters
                print(k)
                kmeans = KMeans(n_clusters=k)
                q, mod = divmod(k, 2)
                
                vis = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q-1][mod])
                vis.fit(data)
            
            plt.plot(means, inertias, 'o-')
            plt.xlabel("Number of Clusters")
            plt.ylabel("Inertia")
            plt.grid(True)
            plt.show()

        data = workingdf[['GR', 'RHOB']]
        kmeans = KMeans(n_clusters=10)
        visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
        visualizer.fit(data)
        visualizer.show()

        from sklearn.metrics import silhouette_score

        def optimise_k_means_sillouette(data, max_k):
            means = []
            silhouette_avg = []
            
            range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
            silhouette_avg = []
            for num_clusters in range_n_clusters:

                # initialise kmeans
                kmeans = KMeans(n_clusters=num_clusters)
                kmeans.fit(data)
                cluster_labels = kmeans.labels_

                # silhouette score
                silhouette_avg.append(silhouette_score(data, cluster_labels))
            plt.plot(range_n_clusters,silhouette_avg, 'bo-')
            plt.xlabel("Number of Clusters") 
            plt.ylabel('Silhouette Score') 
            plt.title('Silhouette Analysis For Kmeans', fontsize=14, fontweight='bold')
            plt.show()

        optimise_k_means_sillouette(workingdf[['GR', 'RHOB', 'NPHI', 'DTC']], 5)

        workingdf.dropna(inplace=True)

        optimise_k_means(workingdf[['GR', 'RHOB', 'NPHI', 'DTC']], 30)

        # Create the KMeans model with the selected number of clusters
        kmeans = KMeans(n_clusters=5)

        # Fit the model to our dataset
        kmeans.fit(workingdf[['GR', 'RHOB', 'NPHI', 'DTC']])

        # Assign the data back to the workingdf
        workingdf['KMeans'] = kmeans.labels_

        # Create the gmm model with the selected number of clusters/components
        gmm = GaussianMixture(n_components=5)

        # Fit the model to our dataset
        gmm.fit(workingdf[['GR', 'RHOB', 'NPHI', 'DTC']])

        # Predict the labels
        gmm_labels = gmm.predict(workingdf[['GR', 'RHOB', 'NPHI', 'DTC']])

        # Assign the labels back to the workingdf
        workingdf['GMM'] = gmm_labels

        dfs_wells, wellnames = well_splitter(workingdf, 'WELL')

        # Setup the curves to plot
        curves_to_plot = ['GR', 'RHOB', 'NPHI', 'DTC',  'LITH_SI', 'KMeans','GMM']
        logarithmic_curves = ['RDEP']
        facies_curve=['KMeans','GMM', 'LITH_SI']

        # Create plot by passing in the relevant well index number
        well = 4
        cmap_facies = create_plot(wellnames[well], 
                    dfs_wells[well], 
                    curves_to_plot, 
                    dfs_wells[well]['DEPTH_MD'], 
                    logarithmic_curves, facies_curve)
    else: 
        st.write("Sorry, unable to generate output. Data format does not match our requirements.")


def Random_Forest_For_Lithology_Classification(df):

    required_columns = ['LITH']
        #Create the figure
    if all(col in df.columns for col in required_columns):  

        g = sns.FacetGrid(df, col='LITH', col_wrap=4)
        g.map(sns.scatterplot, 'NPHI', 'RHOB', alpha=0.5)
        g.set(xlim=(-0.15, 1))
        g.set(ylim=(3, 1))

        g = sns.FacetGrid(df, col='LITH', col_wrap=4)
        g.map(sns.scatterplot, 'DTC', 'RHOB', alpha=0.5)
        g.set(xlim=(40, 240))
        g.set(ylim=(3, 1))

        df.dropna(inplace=True)

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.ensemble import RandomForestClassifier

        # Select inputs and target
        X = df[['RDEP', 'RHOB', 'GR', 'NPHI', 'PEF', 'DTC']]
        y = df['LITH']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        clf = RandomForestClassifier()

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        cf_matrix = confusion_matrix(y_test, y_pred)

        labels = ['Shale', 'Sandstone', 'Sandstone/Shale', 'Limestone', 'Tuff',
            'Marl', 'Anhydrite', 'Dolomite', 'Chalk', 'Coal', 'Halite']
        labels.sort()

        
        fig = plt.figure(figsize=(10,10))
        ax = sns.heatmap(cf_matrix, annot=True, cmap='Reds', fmt='.0f',
                        xticklabels=labels, 
                        yticklabels = labels)

        ax.set_title('Seaborn Confusion Matrix with labels\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ')

        st.write(f"Accuracy: {accuracy}")

        st.pyplot(fig)
    else: 
        st.write("Sorry, unable to generate output. Data format does not match our requirements.")

def Random_Forest_for_LinearRegression(df):



    required_columns = ['WELL','DEPTH', 'RHOB', 'GR', 'NPHI', 'PEF', 'DT']
        #Create the figure
    if all(col in df.columns for col in required_columns): 

        df = df[['WELL', 'DEPTH', 'RHOB', 'GR', 'NPHI', 'PEF', 'DT']].copy()

            # Training Wells
        training_wells = ['15/9-F-11 B', '15/9-F-11 A', '15/9-F-1 A']

        # Test Well
        test_well = ['15/9-F-1 B']

        train_val_df = df[df['WELL'].isin(training_wells)].copy()
        test_df = df[df['WELL'].isin(test_well)].copy()

        train_val_df.dropna(inplace=True)
        test_df.dropna(inplace=True)

        from sklearn.model_selection import train_test_split
        from sklearn import metrics
        from sklearn.ensemble import RandomForestRegressor

        X = train_val_df[['RHOB', 'GR', 'NPHI', 'PEF']]
        y = train_val_df['DT']

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        regr = RandomForestRegressor()

        regr.fit(X_train, y_train)

        y_pred = regr.predict(X_val)

        metrics.mean_absolute_error(y_val, y_pred)

        mse = metrics.mean_squared_error(y_val, y_pred)
        rmse = mse**0.5 

        fig,ax = plt.subplots()
        plt.scatter(y_val, y_pred)
        plt.xlim(40, 140)
        plt.ylim(40, 140)
        plt.ylabel('Predicted DT')
        plt.xlabel('Actual DT')
        plt.plot([40,140], [40,140], 'black') #1 to 1 line
        st.pyplot(fig)
        

        test_well_x = test_df[['RHOB', 'GR', 'NPHI', 'PEF']]

        test_df['TEST_DT'] = regr.predict(test_well_x)

        fig,ax = plt.subplots()
        plt.scatter(test_df['DT'], test_df['TEST_DT'])
        plt.xlim(40, 140)
        plt.ylim(40, 140)
        plt.ylabel('Predicted DT')
        plt.xlabel('Actual DT')
        plt.plot([40,140], [40,140], 'black') #1 to 1 line
        st.pyplot(fig)


        fig, ax = plt.subplots(figsize=(15, 5))
        plt.plot(test_df['DEPTH'], test_df['DT'], label='Actual DT')
        plt.plot(test_df['DEPTH'], test_df['TEST_DT'], label='Predicted DT')
        plt.xlabel('Depth (m)')
        plt.ylabel('DT')
        plt.ylim(40, 140)
        plt.legend()
        plt.grid()
        st.pyplot(fig)
    else: 
        st.write("Sorry, unable to generate output. Data format does not match our requirements.")

def Isolation_Forest_for_Auto_Outlier_Detector(df):
 

    required_columns = ['RHOB', 'GR', 'NPHI', 'PEF','CALI', 'DTC']
        #Create the figure
    if all(col in df.columns for col in required_columns): 


        from sklearn.ensemble import IsolationForest

        df = df.dropna()

        anomaly_inputs = ['NPHI', 'RHOB', 'GR', 'CALI', 'PEF', 'DTC']
        
        
        model_IF = IsolationForest(contamination=0.1, random_state=42)
        model_IF.fit(df[anomaly_inputs])
        df['anomaly_scores'] = model_IF.decision_function(df[anomaly_inputs])
        df['anomaly'] = model_IF.predict(df[anomaly_inputs])

        def outlier_plot(data, outlier_method_name, x_var, y_var, 
                    xaxis_limits=[0,1], yaxis_limits=[0,1]):
        
            st.write(f'Outlier Method: {outlier_method_name}')
            
            method = f'{outlier_method_name}_anomaly'
            
            st.write(f"Number of anomalous values {len(data[data['anomaly']==-1])}")
            st.write(f"Number of non anomalous values  {len(data[data['anomaly']== 1])}")
            st.write(f'Total Number of Values: {len(data)}')
            
            g = sns.FacetGrid(data, col='anomaly', height=4, hue='anomaly', hue_order=[1,-1])
            g.map(sns.scatterplot, x_var, y_var)
            g.fig.suptitle(f'Outlier Method: {outlier_method_name}', y=1.10, fontweight='bold')
            g.set(xlim=xaxis_limits, ylim=yaxis_limits)
            axes = g.axes.flatten()
            axes[0].set_title(f"Outliers\n{len(data[data['anomaly']== -1])} points")
            axes[1].set_title(f"Inliers\n {len(data[data['anomaly']==  1])} points")
            st.pyplot(g)

        outlier_plot(df, 'Isolation Forest', 'NPHI', 'RHOB', [0, 0.8], [3, 1.5]);
        palette = ['#ff7f0e', '#1f77b4']
        pairplot = sns.pairplot(df, vars=anomaly_inputs, hue='anomaly', palette=palette)
        st.pyplot(pairplot)
    else: 
        st.write("Sorry, unable to generate output. Data format does not match our requirements.")






def SIDEBAR1():
   
    #Sidebar navigation
    st.sidebar.title('Navigation')
    options = st.sidebar.radio('Select what you want to display:', ['Min/Max', 'Log Plots','Interactive Plots'])
    # Navigation options
    if options == 'Min/Max':
        min_max(df)
    elif options == 'Log Plots':
        logplots(df)   
    elif options == 'Interactive Plots':
        interactive_plot()
    

    
def SIDEBAR2():
    # Sidebar setup
   
    #Sidebar navigation
    st.sidebar.title('Navigation')
    options = st.sidebar.radio('Select what you want to display:', ['Missing Values', 'Outliers', 'Normalization'])
    # Navigation options
    if options == 'Missing Values':
        missingvalues(df)
    elif options == 'Outliers':
        outlier(df)   
    elif options == 'Normalization':
        normalization(df)
    
def SIDEBAR3():
    # Sidebar setup
  
    #Sidebar navigation
    st.sidebar.title('Navigation')
    options = st.sidebar.radio('Select what you want to display:', ['Poro-Perm General', 'Poro-Perm Detailed', 'Semi-Log','Box-Plots'])
    # Navigation options
    if options == 'Poro-Perm General':
        poro(df)
    elif options == 'Poro-Perm Detailed':
        poroperm_detailed(df)  
    elif options == 'Semi-Log':
        semilog(df)
    elif options == 'Box-Plots':
        boxplots(df)    
    
def SIDEBAR4():
    # Sidebar setup
    
    st.sidebar.title('Navigation')
    options = st.sidebar.radio('Select what you want to display:', [
        'Unsupervised Clustering for Lithofacies','Random Forest for Lithology Classification',
                                   'Random Forest for Regression','Isolation Forest for Auto Outlier Detection'])
    if options == 'Unsupervised Clustering for Lithofacies':
        Unsupervised_Clustering_for_Lithofacies(df)
    elif options == 'Random Forest for Lithology Classification':
        Random_Forest_For_Lithology_Classification(df)  
    elif options == 'Random Forest for Regression':
        Random_Forest_for_LinearRegression(df)
    elif options == 'Isolation Forest for Auto Outlier Detection':
        Isolation_Forest_for_Auto_Outlier_Detector(df)   




































# Add a title and intro text
st.title('CARBON SENSE AI')
st.text('This is a web app that allows exploration, engineering and applciation of machine learning models on Petrophysics Data')

# Load the image
st.image('C:/Users/poude/OneDrive/Desktop/HOMEPAGE.jpeg', width=500)


st.sidebar.image('C:/Users/poude/OneDrive/Desktop/Green and Cream Leaves Landscaping Logo.png', width=150)

# Display the image

upload_file = st.sidebar.file_uploader('Upload a WellLog data')



# Check if file has been uploaded
if upload_file is not None:
    df = pd.read_csv(upload_file,header = 0, skiprows =[1])









# Create a dropdown button in the body of the website
selected_option = st.selectbox('What would you like to do with your data?', ['Data Exploration & Visualization', 
                                                 'Data Preprocessing', 'Feature Extraction & Engineering','Machine Learning Modeling'])











# Display the corresponding buttons based on the selected option
if selected_option == 'Data Exploration & Visualization':
    SIDEBAR1()    
elif selected_option == 'Data Preprocessing':
    SIDEBAR2()
elif selected_option == 'Feature Extraction & Engineering':    
    SIDEBAR3()
elif selected_option == 'Machine Learning Modeling':    
    SIDEBAR4()






