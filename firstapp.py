#Import the required Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lasio
from io import StringIO

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
from yellowbrick.cluster import SilhouetteVisualizer

st.set_page_config(layout="wide")

# Functions for each of the pages

def interactive_plot():
    col1, col2, col3 = st.columns(3)

    x_axis_val = col1.selectbox('Select the X-axis', options=df.columns)
    x_scale = col1.selectbox('Select the X-axis scale', options=['linear', 'log'])

    y_axis_val = col2.selectbox('Select the Y-axis', options=df.columns)
    y_scale = col2.selectbox('Select the Y-axis scale', options=['linear', 'log'])

    plot = px.scatter(df, x=x_axis_val, y=y_axis_val)

    if x_scale == 'log':
        plot.update_xaxes(type='log')
    if y_scale == 'log':
        plot.update_yaxes(type='log')

    col3.plotly_chart(plot, use_container_width=True)

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




                # Assuming df is defined elsewhere

    # Filter columns that end with "_E"
    cols_to_plot = [col for col in df.columns if col.endswith("_E")]

    num_cols = len(cols_to_plot)
    cols = 2
    rows = (num_cols + 1) // cols  # Add 1 to ensure at least 1 row if num_cols is less than cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))

    for i, feature in enumerate(cols_to_plot):
        row_index = i // cols
        col_index = i % cols

        ax = axes[row_index, col_index] if rows > 1 else axes[col_index]
        df[feature].hist(bins=10, ax=ax, facecolor='green', alpha=0.6)

        ax.set_title(feature + " Distribution")
        ax.set_axisbelow(True)
        ax.grid(color='whitesmoke')

    plt.tight_layout()
    st.pyplot(fig)



def logplots(df):


        df.replace(to_replace=r'^[a-zA-Z]+$', value=-999.00, regex=True, inplace=True)

        well_nan = df.notnull() * 1

        fig, ax = plt.subplots(figsize=(15,10))

        #Set up the plot axes
        ax1 = plt.subplot2grid((1,6), (0,0), rowspan=1, colspan = 1)
        ax2 = plt.subplot2grid((1,6), (0,1), rowspan=1, colspan = 1, sharey = ax1)
        ax3 = plt.subplot2grid((1,6), (0,2), rowspan=1, colspan = 1, sharey = ax1)
        ax4 = plt.subplot2grid((1,6), (0,3), rowspan=1, colspan = 1, sharey = ax1)
        ax5 = ax3.twiny() #Twins the y-axis for the density track with the neutron track
        ax6 = plt.subplot2grid((1,6), (0,4), rowspan=1, colspan = 1, sharey = ax1)
        ax7 = ax2.twiny()
        ax8 = ax2.twiny()

        # As our curve scales will be detached from the top of the track,
        # this code adds the top border back in without dealing with splines
        ax10 = ax1.twiny()
        ax10.xaxis.set_visible(False)
        ax11 = ax2.twiny()
        ax11.xaxis.set_visible(False)
        ax12 = ax3.twiny()
        ax12.xaxis.set_visible(False)
        ax13 = ax4.twiny()
        ax13.xaxis.set_visible(False)
        ax14 = ax6.twiny()
        ax14.xaxis.set_visible(False)

        # Gamma Ray track
        if 'GR_E' in df.columns:
            ax1.plot(df["GR_E"], df["DEPTH"], color = "green", linewidth = 0.5)
            ax1.set_xlabel("Gamma")
            ax1.xaxis.label.set_color("green")
        
            ax1.set_ylabel("Depth (ft)")
            ax1.tick_params(axis='x', colors="green")
            ax1.spines["top"].set_edgecolor("green")
            ax1.title.set_color('green')
      

        # Resistivity track
        if 'RESD_E' in df.columns:
            ax2.plot("RESD_E", "DEPTH", data = df, color = "red", linewidth = 0.5)
            ax2.set_xlabel("Resistivity - Deep")
            
            ax2.xaxis.label.set_color("red")
            ax2.tick_params(axis='x', colors="red")
            ax2.spines["top"].set_edgecolor("red")
        
            ax2.semilogx()

        # Density track
        if 'RHOB_E' in df.columns:
            ax3.plot("RHOB_E", "DEPTH", data = df, color = "red", linewidth = 0.5)
            ax3.set_xlabel("Density")
            ax3.set_xlim(df['RHOB_E'].min(), df['RHOB_E'].max())
            ax3.xaxis.label.set_color("red")
            ax3.tick_params(axis='x', colors="red")
            ax3.spines["top"].set_edgecolor("red")
      

        # Sonic track
        if 'DT_E' in df.columns:
            ax4.plot("DT_E", "DEPTH", data = df, color = "purple", linewidth = 0.5)
            ax4.set_xlabel("Sonic")
            
            ax4.xaxis.label.set_color("purple")
            ax4.tick_params(axis='x', colors="purple")
            ax4.spines["top"].set_edgecolor("purple")

            # Neutron track placed ontop of density track
        if 'NPHI_E' in df.columns:
            ax5.plot("NPHI_E", "DEPTH", data = df, color = "blue", linewidth = 0.5)
            ax5.set_xlabel('Neutron')
            ax5.xaxis.label.set_color("blue")
            ax5.set_xlim(df['NPHI_E'].max(), df['NPHI_E'].min())
        
            ax5.tick_params(axis='x', colors="blue")
            ax5.spines["top"].set_position(("axes", 1.08))
            ax5.spines["top"].set_visible(True)
            ax5.spines["top"].set_edgecolor("blue")
        

        # Caliper track
        if 'CALI_E' in df.columns:
            ax6.plot("CALI_E", "DEPTH", data = df, color = "black", linewidth = 0.5)
            ax6.set_xlabel("Caliper")
        
            ax6.xaxis.label.set_color("black")
            ax6.tick_params(axis='x', colors="black")
            ax6.spines["top"].set_edgecolor("black")
            ax6.fill_betweenx(well_nan.index, 8.5, df["CALI_E"], facecolor='yellow')
        

        # Resistivity track - Curve 2
        if 'RESM_E' in df.columns:
            ax7.plot("RESM_E", "DEPTH", data = df, color = "green", linewidth = 0.5)
            ax7.set_xlabel("Resistivity - Med")
        
            ax7.xaxis.label.set_color("green")
            ax7.spines["top"].set_position(("axes", 1.16))
            ax7.spines["top"].set_visible(True)
            ax7.tick_params(axis='x', colors="green")
            ax7.spines["top"].set_edgecolor("green")
            
            ax7.semilogx()

        # Resistivity track - Curve 2
        if 'RESS_E' in df.columns:
            ax8.plot("RESS_E", "DEPTH", data = df, color = "blue", linewidth = 0.5)
            ax8.set_xlabel("Resistivity - Sha")
        
            ax8.xaxis.label.set_color("blue")
            ax8.spines["top"].set_position(("axes", 1.09))
            ax8.spines["top"].set_visible(True)
            ax8.tick_params(axis='x', colors="blue")
            ax8.spines["top"].set_edgecolor("blue")
            
            ax8.semilogx()

        x1=df['RHOB_E']
        x2=df['NPHI_E']

        x = np.array(ax3.get_xlim())
        z = np.array(ax5.get_xlim())

        nz=((x2-np.max(z))/(np.min(z)-np.max(z)))*(np.max(x)-np.min(x))+np.min(x)
       
        ax3.fill_betweenx(df['DEPTH'], x1, nz, where=x1>=nz, interpolate=True, color='green')
        ax3.fill_betweenx(df['DEPTH'], x1, nz, where=x1<=nz, interpolate=True, color='yellow')       

        # Common functions for setting up the plot can be extracted into
        # a for loop. This saves repeating code.
        for ax in [ax1, ax2, ax3, ax4, ax6]:
           
            ax.grid(which='major', color='lightgrey', linestyle='-')
            ax.xaxis.set_ticks_position("top")
            ax.xaxis.set_label_position("top")
            ax.spines["top"].set_position(("axes", 1.02))
            
            
        for ax in [ax2, ax3, ax4, ax6]:
            plt.setp(ax.get_yticklabels(), visible = False)
            
        plt.tight_layout()
        fig.subplots_adjust(wspace = 0.15)
        
        st.pyplot(fig)











        
        st.header("Adjusted Log Plot")





                # Calculate the 95th percentile for "Gamma" and "Caliper"
        gamma_95th_percentile = np.nanpercentile(df["GR_E"], 95)
        caliper_95th_percentile = np.nanpercentile(df["CALI_E"], 95)
        



        df.replace(to_replace=r'^[a-zA-Z]+$', value=-999.00, regex=True, inplace=True)

        well_nan = df.notnull() * 1

        fig, ax = plt.subplots(figsize=(15,10))

        #Set up the plot axes
        ax1 = plt.subplot2grid((1,6), (0,0), rowspan=1, colspan = 1)
        ax2 = plt.subplot2grid((1,6), (0,1), rowspan=1, colspan = 1, sharey = ax1)
        ax3 = plt.subplot2grid((1,6), (0,2), rowspan=1, colspan = 1, sharey = ax1)
        ax4 = plt.subplot2grid((1,6), (0,3), rowspan=1, colspan = 1, sharey = ax1)
        ax5 = ax3.twiny() #Twins the y-axis for the density track with the neutron track
        ax6 = plt.subplot2grid((1,6), (0,4), rowspan=1, colspan = 1, sharey = ax1)
        ax7 = ax2.twiny()
        ax8 = ax2.twiny()

        # As our curve scales will be detached from the top of the track,
        # this code adds the top border back in without dealing with splines
        ax10 = ax1.twiny()
        ax10.xaxis.set_visible(False)
        ax11 = ax2.twiny()
        ax11.xaxis.set_visible(False)
        ax12 = ax3.twiny()
        ax12.xaxis.set_visible(False)
        ax13 = ax4.twiny()
        ax13.xaxis.set_visible(False)
        ax14 = ax6.twiny()
        ax14.xaxis.set_visible(False)

        # Gamma Ray track
        if 'GR_E' in df.columns:
            gamma_values = df["GR_E"]
            gamma_values[gamma_values >= gamma_95th_percentile] = np.nan  # Set values >= 95th percentile to NaN
            ax1.plot(gamma_values, df["DEPTH"], color="green", linewidth=0.5)
            ax1.set_xlabel("Gamma")
            ax1.xaxis.label.set_color("green")
            ax1.set_ylabel("Depth (ft)")
            ax1.tick_params(axis='x', colors="green")
            ax1.spines["top"].set_edgecolor("green")
            ax1.title.set_color('green')
           
      

        # Resistivity track
        if 'RESD_E' in df.columns:
            ax2.plot("RESD_E", "DEPTH", data = df, color = "red", linewidth = 0.5)
            ax2.set_xlabel("Resistivity - Deep")
            
            ax2.xaxis.label.set_color("red")
            ax2.tick_params(axis='x', colors="red")
            ax2.spines["top"].set_edgecolor("red")
           
            ax2.semilogx()

        # Density track
        if 'RHOB_E' in df.columns:
            ax3.plot("RHOB_E", "DEPTH", data = df, color = "red", linewidth = 0.5)
            ax3.set_xlabel("Density")
            
            ax3.set_xlim(df['RHOB_E'].min(), df['RHOB_E'].max())
            ax3.xaxis.label.set_color("red")
            ax3.tick_params(axis='x', colors="red")
            ax3.spines["top"].set_edgecolor("red")
      

        # Sonic track
        if 'DT_E' in df.columns:
            ax4.plot("DT_E", "DEPTH", data = df, color = "purple", linewidth = 0.5)
            ax4.set_xlabel("Sonic")
            
            ax4.xaxis.label.set_color("purple")
            ax4.tick_params(axis='x', colors="purple")
            ax4.spines["top"].set_edgecolor("purple")

            # Neutron track placed ontop of density track
        if 'NPHI_E' in df.columns:
            ax5.plot("NPHI_E", "DEPTH", data = df, color = "blue", linewidth = 0.5)
            ax5.set_xlabel('Neutron')
            ax5.xaxis.label.set_color("blue")
            ax5.set_xlim(df['NPHI_E'].max(), df['NPHI_E'].min())
        
            ax5.tick_params(axis='x', colors="blue")
            ax5.spines["top"].set_position(("axes", 1.08))
            ax5.spines["top"].set_visible(True)
            ax5.spines["top"].set_edgecolor("blue")
        

        # Caliper track
    # Caliper track
        if 'CALI_E' in df.columns:
            caliper_values = df["CALI_E"]
            caliper_values[caliper_values >= caliper_95th_percentile] = np.nan  # Set values >= 95th percentile to NaN
            ax6.plot(caliper_values, df["DEPTH"], color="black", linewidth=0.5)
            ax6.set_xlabel("Caliper")
            ax6.xaxis.label.set_color("black")
            ax6.tick_params(axis='x', colors="black")
            ax6.spines["top"].set_edgecolor("black")
            ax6.fill_betweenx(df['DEPTH'], 8.5, df["CALI_E"], facecolor='yellow')
        

        # Resistivity track - Curve 2
        if 'RESM_E' in df.columns:
            ax7.plot("RESM_E", "DEPTH", data = df, color = "green", linewidth = 0.5)
            ax7.set_xlabel("Resistivity - Med")
        
            ax7.xaxis.label.set_color("green")
            ax7.spines["top"].set_position(("axes", 1.16))
            ax7.spines["top"].set_visible(True)
            ax7.tick_params(axis='x', colors="green")
            ax7.spines["top"].set_edgecolor("green")
            
            ax7.semilogx()

        # Resistivity track - Curve 2
        if 'RESS_E' in df.columns:
            ax8.plot("RESS_E", "DEPTH", data = df, color = "blue", linewidth = 0.5)
            ax8.set_xlabel("Resistivity - Sha")
        
            ax8.xaxis.label.set_color("blue")
            ax8.spines["top"].set_position(("axes", 1.09))
            ax8.spines["top"].set_visible(True)
            ax8.tick_params(axis='x', colors="blue")
            ax8.spines["top"].set_edgecolor("blue")
            
            ax8.semilogx()

        

                # Adding in neutron density shading
        x1=df['RHOB_E']
        x2=df['NPHI_E']

        x = np.array(ax3.get_xlim())
        z = np.array(ax5.get_xlim())

        nz=((x2-np.max(z))/(np.min(z)-np.max(z)))*(np.max(x)-np.min(x))+np.min(x)
       
        ax3.fill_betweenx(df['DEPTH'], x1, nz, where=x1>=nz, interpolate=True, color='green')
        ax3.fill_betweenx(df['DEPTH'], x1, nz, where=x1<=nz, interpolate=True, color='yellow')





        # Common functions for setting up the plot can be extracted into
        # a for loop. This saves repeating code.
        for ax in [ax1, ax2, ax3, ax4, ax6]:
           
            ax.grid(which='major', color='lightgrey', linestyle='-')
            ax.xaxis.set_ticks_position("top")
            ax.xaxis.set_label_position("top")
            ax.spines["top"].set_position(("axes", 1.02))
            
            
        for ax in [ax2, ax3, ax4, ax6]:
            plt.setp(ax.get_yticklabels(), visible = False)
            
        plt.tight_layout()
        fig.subplots_adjust(wspace = 0.15)
        
        st.pyplot(fig)


        
    
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
    
    fig = plt.gcf()
    # Display the plot in Streamlit
    st.pyplot(fig)


def outlier(df):

    
    df_e = pd.DataFrame()  # Initialize DataFrame for columns ending with "_E"
    df_f = pd.DataFrame()  # Initialize DataFrame for columns ending with "_F"
    df_s = pd.DataFrame()  # Initialize DataFrame for columns ending with "_S"
    df_u = pd.DataFrame()  # Initialize DataFrame for columns ending with "_U"
    
    

    for col in df.columns:
        if col.endswith('_E'):
            df_e[col] = df[col]  # Add column to df_e
        elif col.endswith('_F'):
            df_f[col] = df[col]  # Add column to df_f
        elif col.endswith('_S'):
            df_s[col] = df[col]  # Add column to df_s
        elif col.endswith('_U'):
            df_u[col] = df[col]  # Add column to df_u
    
    if not df_e.empty:
        df=df_e


    
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_columns) == len(df.columns):
    
        
        df.dropna(inplace=True)
        
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
        fig = plt.gcf()
        st.pyplot(fig)
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

    df = df.assign(WELL=1)


    required_columns = [ "RHOB_E", "GR_E", "NPHI_E", "DT_E"]
        #Create the figure
    if all(col in df.columns for col in required_columns):

        workingdf = df[[ "WELL","DEPTH", "RHOB_E", "GR_E", "NPHI_E", "DT_E"]].copy()
       
        def create_plot(wellname, dataframe, curves_to_plot, depth_curve, facies_curves=[]):
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
                
                

            
            plt.tight_layout()
            st.pyplot(fig)
            
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
            
            return wells_as_dfs, wells_wellnames
        
            
        

        workingdf.dropna(inplace =True)        

        
           
        data = workingdf[['GR_E', 'RHOB_E']]

             # Create a sidebar for user input
        st.sidebar.header('KMeans Silhouette Analysis')
        num_clusters = st.sidebar.slider('Select number of clusters:', 2, 20, 5)
        
        # Fit KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters)
        visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
        
        # Fit the data to the visualizer
        visualizer.fit(data)
        
        # Display the visualizer
        st.pyplot()

        # Display the visualizer
        st.pyplot(visualizer.show())
        plt.savefig("silhouette_plot.png")

        


        def optimise_k_means_sillouette(data, max_k):
            range_n_clusters = list(range(2, max_k + 1))  # Update to include max_k
            silhouette_avg = []

            for num_clusters in range_n_clusters:
                # Initialise kmeans
                kmeans = KMeans(n_clusters=num_clusters)
                kmeans.fit(data)
                cluster_labels = kmeans.labels_

                # Silhouette score
                silhouette_avg.append(silhouette_score(data, cluster_labels))

            # Plot the silhouette scores
            plt.plot(range_n_clusters, silhouette_avg, 'bo-')
            plt.xlabel("Number of Clusters")
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Analysis For KMeans')
            plt.grid(True)

            # Display the plot
            st.pyplot()

        
        # Create a sidebar for user input
        st.sidebar.header('Optimize KMeans Silhouette Analysis')
        max_clusters = st.sidebar.slider('Select maximum number of clusters:', 2, 20, 5)

        # Call the function within the Streamlit app
        optimise_k_means_sillouette(workingdf[['GR_E', 'RHOB_E', 'NPHI_E', 'DT_E']], max_clusters)


               

        # Create the KMeans model with the selected number of clusters
        kmeans = KMeans(n_clusters=5)

        # Fit the model to our dataset
        kmeans.fit(workingdf[['GR_E', 'RHOB_E', 'NPHI_E', 'DT_E']])

        # Assign the data back to the workingdf
        workingdf['KMeans'] = kmeans.labels_

        # Create the gmm model with the selected number of clusters/components
        gmm = GaussianMixture(n_components=5)

        # Fit the model to our dataset
        gmm.fit(workingdf[['GR_E', 'RHOB_E', 'NPHI_E', 'DTC_E']])

        # Predict the labels
        gmm_labels = gmm.predict(workingdf[['GR_E', 'RHOB_E', 'NPHI_E', 'DT_E']])

        # Assign the labels back to the workingdf
        workingdf['GMM'] = gmm_labels

        dfs_wells, wellnames = well_splitter(workingdf, 'WELL')

        # Setup the curves to plot
        curves_to_plot = ['GR_E', 'RHOB_E', 'NPHI_E', 'DT_E', 'KMeans','GMM']
        
        facies_curve=['KMeans','GMM']

        # Create plot by passing in the relevant well index number
        well = 0
        cmap_facies = create_plot(wellnames[well], 
                    dfs_wells[well], 
                    curves_to_plot, 
                    dfs_wells[well]['DEPTH'], 
                     facies_curve)
    


        fig, ax = plt.subplots(figsize=(20,10))
        ax1 = plt.subplot2grid((1,3), (0,0))
        ax1.scatter(dfs_wells[well]['NPHI_E'], dfs_wells[well]['RHOB_E'], c=dfs_wells[well]['KMeans'], s=8, cmap=cmap_facies)
        ax1.set_title('KMeans', fontsize=22, y=1.05)

        ax2 = plt.subplot2grid((1,3), (0,1))
        ax2.scatter(dfs_wells[well]['NPHI_E'], dfs_wells[well]['RHOB_E'], c=dfs_wells[well]['GMM'], s=8)
        ax2.set_title('GMM', fontsize=22, y=1.05)

       

        for ax in [ax1, ax2]:
            ax.set_xlim(0, 0.7)
            ax.set_ylim(3, 1.5)
            ax.set_ylabel('RHOB', fontsize=18, labelpad=30)
            ax.set_xlabel('NPHI', fontsize=18, labelpad=30)
            ax.grid()
            ax.set_axisbelow(True)

            ax.tick_params(axis='both', labelsize=14)
        plt.tight_layout()
        
        sns.pairplot(dfs_wells[1], vars=['GR_E', 'RHOB_E','NPHI_E', 'DT_E'], hue='KMeans', palette='Dark2',
             diag_kind='kde', plot_kws = {'s': 15, 'marker':'o', 'alpha':1})
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
 

    required_columns = ['RHOB_E', 'GR_E', 'NPHI_E','CALI_E', 'DT_E']
        #Create the figure
    if all(col in df.columns for col in required_columns): 


        from sklearn.ensemble import IsolationForest

        df = df.dropna()

        anomaly_inputs = ['NPHI_E', 'GR_E', 'CALI_E', 'DT_E']
        
        
        model_IF = IsolationForest(contamination=0.1, random_state=42)
        model_IF.fit(df[anomaly_inputs])
        df['anomaly_scores'] = model_IF.decision_function(df[anomaly_inputs])
        df['anomaly'] = model_IF.predict(df[anomaly_inputs])

        def outlier_plot(data, outlier_method_name, x_var, y_var, 
                   ):
        
            st.write(f'Outlier Method: {outlier_method_name}')
            
            method = f'{outlier_method_name}_anomaly'
            
            st.write(f"Number of anomalous values {len(data[data['anomaly']==-1])}")
            st.write(f"Number of non anomalous values  {len(data[data['anomaly']== 1])}")
            st.write(f'Total Number of Values: {len(data)}')
            
            g = sns.FacetGrid(data, col='anomaly', height=4, hue='anomaly', hue_order=[1,-1])
            g.map(sns.scatterplot, x_var, y_var)
            g.fig.suptitle(f'Outlier Method: {outlier_method_name}', y=1.10, fontweight='bold')
            
            axes = g.axes.flatten()
            axes[0].set_title(f"Outliers\n{len(data[data['anomaly']== -1])} points")
            axes[1].set_title(f"Inliers\n {len(data[data['anomaly']==  1])} points")
            st.pyplot(g)

        outlier_plot(df, 'Isolation Forest', 'NPHI_E', 'RHOB_E');
        palette = ['#ff7f0e', '#1f77b4']
        pairplot = sns.pairplot(df, vars=anomaly_inputs, hue='anomaly', palette=palette)
        st.pyplot(pairplot)
    else: 
        st.write("Sorry, unable to generate output. Data format does not match our requirements.")






def SIDEBAR1(df):
   
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
    

    
def SIDEBAR2(df):
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
    
def SIDEBAR3(df):
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
    
def SIDEBAR4(df):
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
st.text('This is a web app that allows exploration, engineering and application of machine learning models on Petrophysics Data')

# Load the image
st.image('HOMEPAGE.jpeg', width=500)


st.sidebar.image('Green and Cream Leaves Landscaping Logo.png', width=150)

# Display the image

df = None


file_type = st.sidebar.radio("Select file type:", ("LAS", "CSV"))



if file_type == "LAS":
    las_file = st.file_uploader("Upload LAS file", type=["las"])
    if las_file is not None:
        try:
                bytes_data = las_file.read()
                str_io = StringIO(bytes_data.decode('Windows-1252'))
                las_file = lasio.read(str_io)
                df = las_file.df()
                df['DEPTH'] = df.index

        except UnicodeDecodeError as e:
                st.error(f"error loading log.las: {e}")
                df=None
    






        
elif file_type == "CSV":
    csv_file = st.file_uploader("Upload CSV file", type=["csv"])
    if csv_file is not None:
        st.success("CSV file uploaded successfully!")
        # Process CSV file
        df = pd.read_csv(csv_file)











# Create a dropdown button in the body of the website
selected_option = st.selectbox('What would you like to do with your data?', ['Data Exploration & Visualization', 
                                                 'Data Preprocessing', 'Feature Extraction & Engineering','Machine Learning Modeling'])









if df is not None:

    # Display the corresponding buttons based on the selected option
    if selected_option == 'Data Exploration & Visualization':
        SIDEBAR1(df)    
    elif selected_option == 'Data Preprocessing':
        SIDEBAR2(df)
    elif selected_option == 'Feature Extraction & Engineering':    
        SIDEBAR3(df)
    elif selected_option == 'Machine Learning Modeling':    
        SIDEBAR4(df)






