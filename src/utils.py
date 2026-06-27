import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from adjustText import adjust_text

class ForensicVisualizer:
    """
    Clase de utilidades para generar visualizaciones avanzadas y estandarizadas 
    del análisis forense de redes y anomalías.
    """
    
    @staticmethod
    def plot_kmeans_roles(df_model: pd.DataFrame, top_n: int = 12):
        """Genera el Bubble Chart con escala Symlog para el Caso de Uso 1 (Grafos)."""
        plt.figure(figsize=(14, 9))
        
        sns.scatterplot(
            data=df_model, 
            x='Betweenness', 
            y='PageRank', 
            hue='Etiqueta_Forense', 
            size='Out_Degree',
            sizes=(50, 600),
            palette='Set1', 
            edgecolor='black',
            alpha=0.8
        )
        
        plt.xscale('symlog', linthresh=0.01)
        plt.yscale('symlog', linthresh=0.01)
        
        textos_a_dibujar = []
        top_nodes = df_model.nlargest(top_n, 'PageRank') 
        
        for _, row in top_nodes.iterrows():
            textos_a_dibujar.append(
                plt.text(row['Betweenness'], row['PageRank'], row['Nodo'], 
                         fontsize=10, fontweight='bold', ha='center', va='center')
            )
            
        adjust_text(
            textos_a_dibujar, 
            arrowprops=dict(arrowstyle='-', color='gray', lw=1.5),
            expand_points=(1.5, 1.5)
        )
        
        plt.title('Clasificación Automática de Roles en la Red (K-Means)', fontsize=14, pad=15)
        plt.xlabel('Centralidad de Intermediación (Betweenness)', fontsize=12)
        plt.ylabel('Poder Jerárquico (PageRank)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_anomaly_scatter(df_anomalies: pd.DataFrame):
        """Genera el gráfico de dispersión temporal vs hora para el Caso de Uso 2."""
        plt.figure(figsize=(14, 7))
        
        sns.scatterplot(
            data=df_anomalies, 
            x='Datetime', 
            y='Hour', 
            hue='Is_Anomaly', 
            palette={1: 'lightgrey', -1: 'red'},
            size='Message_Length', 
            sizes=(20, 500),
            alpha=0.8,
            edgecolor='black'
        )
        
        plt.title('Detección de Anomalías de Comportamiento (Isolation Forest)', fontsize=15, pad=15)
        plt.xlabel('Línea Temporal (Fecha)', fontsize=12)
        plt.ylabel('Hora del Día (0-23h)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Estado del Modelo")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_temporal_density_and_scores(df_anomalies: pd.DataFrame):
        """Genera el panel dual de Z-Score semanal y evolución del Anomaly Score."""
        # 1. Preparación de datos (Oculta al notebook)
        df = df_anomalies.copy()
        df['Week'] = df['Datetime'].dt.isocalendar().week
        df['Year'] = df['Datetime'].dt.year
        df_weekly = df.groupby(['Year', 'Week']).size().reset_index(name='Message_Count')
        df_weekly['Date'] = pd.to_datetime(df_weekly['Year'].astype(str) + df_weekly['Week'].astype(str) + '1', format='%G%V%u')
        df_weekly['Volume_ZScore'] = (df_weekly['Message_Count'] - df_weekly['Message_Count'].mean()) / df_weekly['Message_Count'].std()
        
        df_recent = df[df['Datetime'].dt.year >= 2020].copy().sort_values('Datetime')
        df_recent['Rolling_Score'] = df_recent['Anomaly_Score'].rolling(window=20, min_periods=1).mean()

        # 2. Configuración del panel
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=False)
        
        # Gráfico A
        sns.lineplot(data=df_weekly, x='Date', y='Message_Count', ax=ax1, color='steelblue', linewidth=2, label='Volumen Semanal')
        sns.scatterplot(data=df_weekly[df_weekly['Volume_ZScore'] > 2], x='Date', y='Message_Count', ax=ax1, color='red', s=150, label='Pico Atípico (Z-Score > 2)')
        ax1.set_title('A. Densidad de Comunicaciones y Picos de Actividad (Análisis Z-Score)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Nº de Mensajes', fontsize=11)
        ax1.set_xlabel('')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()
        
        # Gráfico B
        sns.scatterplot(data=df_recent, x='Datetime', y='Anomaly_Score', hue='Is_Anomaly',
                        palette={1: 'lightgrey', -1: 'darkred'}, s=50, alpha=0.7, ax=ax2, edgecolor=None)
        sns.lineplot(data=df_recent, x='Datetime', y='Rolling_Score', ax=ax2, color='black', linewidth=1.5, label='Tendencia de Tensión (Media Móvil)')
        ax2.set_title('B. Evolución de Anomalías de Comportamiento (Isolation Forest Decision Score)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Anomaly Score (Menor = Más Raro)', fontsize=11)
        ax2.set_xlabel('Línea Temporal', fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='lower right')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        
        for label in ax2.get_xticklabels():
            label.set_rotation(45)
            
        plt.tight_layout()
        plt.show()