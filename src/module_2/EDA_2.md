```python
import boto3

# Inicializar el cliente de S3
s3 = boto3.client('s3')

# Parámetros del archivo
bucket_name = 'zrive-ds-data'
file_key = 'groceries/box_builder_dataset/feature_frame.csv'
output_file = 'feature_frame.csv'  # Nombre del archivo donde quieres guardarlo localmente

# Descargar el archivo
s3.download_file(bucket_name, file_key, output_file)

print(f"Archivo descargado exitosamente como {output_file}")



```

    Archivo descargado exitosamente como feature_frame.csv



```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo CSV desde la ruta indicada
file_path = '/home/eduardo_1999/projects/zrive-ds/src/module_2/feature_frame.csv'

df = pd.read_csv(file_path)

# Mostrar las primeras filas del archivo para verificar la carga correcta
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>...</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2020-10-05 17:59:51</td>
      <td>2020-10-05 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>2020-10-05 20:08:53</td>
      <td>2020-10-05 00:00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2020-10-06 08:57:59</td>
      <td>2020-10-06 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808429314180</td>
      <td>3537167515780</td>
      <td>2020-10-06 10:37:05</td>
      <td>2020-10-06 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
# Ver la forma del DataFrame
print(f"El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas")

# Ver la información del DataFrame (incluye tipos de datos y valores nulos)
df.info()

# Resumen estadístico del DataFrame
df.describe(include='all')

```

    El dataset tiene 2880549 filas y 27 columnas
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2880549 entries, 0 to 2880548
    Data columns (total 27 columns):
     #   Column                            Dtype  
    ---  ------                            -----  
     0   variant_id                        int64  
     1   product_type                      object 
     2   order_id                          int64  
     3   user_id                           int64  
     4   created_at                        object 
     5   order_date                        object 
     6   user_order_seq                    int64  
     7   outcome                           float64
     8   ordered_before                    float64
     9   abandoned_before                  float64
     10  active_snoozed                    float64
     11  set_as_regular                    float64
     12  normalised_price                  float64
     13  discount_pct                      float64
     14  vendor                            object 
     15  global_popularity                 float64
     16  count_adults                      float64
     17  count_children                    float64
     18  count_babies                      float64
     19  count_pets                        float64
     20  people_ex_baby                    float64
     21  days_since_purchase_variant_id    float64
     22  avg_days_to_buy_variant_id        float64
     23  std_days_to_buy_variant_id        float64
     24  days_since_purchase_product_type  float64
     25  avg_days_to_buy_product_type      float64
     26  std_days_to_buy_product_type      float64
    dtypes: float64(19), int64(4), object(4)
    memory usage: 593.4+ MB





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>...</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.880549e+06</td>
      <td>2880549</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2880549</td>
      <td>2880549</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>...</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>62</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3446</td>
      <td>149</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>tinspackagedfoods</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2021-03-03 14:42:05</td>
      <td>2021-02-17 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>226474</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>976</td>
      <td>68446</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.401250e+13</td>
      <td>NaN</td>
      <td>2.978388e+12</td>
      <td>3.750025e+12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.289342e+00</td>
      <td>1.153669e-02</td>
      <td>2.113868e-02</td>
      <td>6.092589e-04</td>
      <td>...</td>
      <td>5.492182e-02</td>
      <td>3.538562e-03</td>
      <td>5.134091e-02</td>
      <td>2.072549e+00</td>
      <td>3.312961e+01</td>
      <td>3.523734e+01</td>
      <td>2.645304e+01</td>
      <td>3.143513e+01</td>
      <td>3.088810e+01</td>
      <td>2.594969e+01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.786246e+11</td>
      <td>NaN</td>
      <td>2.446292e+11</td>
      <td>1.775710e+11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.140176e+00</td>
      <td>1.067876e-01</td>
      <td>1.438466e-01</td>
      <td>2.467565e-02</td>
      <td>...</td>
      <td>3.276586e-01</td>
      <td>5.938048e-02</td>
      <td>3.013646e-01</td>
      <td>3.943659e-01</td>
      <td>3.707162e+00</td>
      <td>1.057766e+01</td>
      <td>7.168323e+00</td>
      <td>1.227511e+01</td>
      <td>4.330262e+00</td>
      <td>3.278860e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.361529e+13</td>
      <td>NaN</td>
      <td>2.807986e+12</td>
      <td>3.046041e+12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.414214e+00</td>
      <td>0.000000e+00</td>
      <td>7.000000e+00</td>
      <td>2.828427e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.380354e+13</td>
      <td>NaN</td>
      <td>2.875152e+12</td>
      <td>3.745901e+12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>3.300000e+01</td>
      <td>3.000000e+01</td>
      <td>2.319372e+01</td>
      <td>3.000000e+01</td>
      <td>2.800000e+01</td>
      <td>2.427618e+01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.397325e+13</td>
      <td>NaN</td>
      <td>2.902856e+12</td>
      <td>3.812775e+12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>3.300000e+01</td>
      <td>3.400000e+01</td>
      <td>2.769305e+01</td>
      <td>3.000000e+01</td>
      <td>3.100000e+01</td>
      <td>2.608188e+01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.428495e+13</td>
      <td>NaN</td>
      <td>2.922034e+12</td>
      <td>3.874925e+12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>3.300000e+01</td>
      <td>4.000000e+01</td>
      <td>3.059484e+01</td>
      <td>3.000000e+01</td>
      <td>3.400000e+01</td>
      <td>2.796118e+01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.454300e+13</td>
      <td>NaN</td>
      <td>3.643302e+12</td>
      <td>5.029635e+12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.100000e+01</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>...</td>
      <td>3.000000e+00</td>
      <td>1.000000e+00</td>
      <td>6.000000e+00</td>
      <td>5.000000e+00</td>
      <td>1.480000e+02</td>
      <td>8.400000e+01</td>
      <td>5.868986e+01</td>
      <td>1.480000e+02</td>
      <td>3.950000e+01</td>
      <td>3.564191e+01</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 27 columns</p>
</div>




```python
# Contar los valores nulos por columna
df.isnull().sum()

```




    variant_id                          0
    product_type                        0
    order_id                            0
    user_id                             0
    created_at                          0
    order_date                          0
    user_order_seq                      0
    outcome                             0
    ordered_before                      0
    abandoned_before                    0
    active_snoozed                      0
    set_as_regular                      0
    normalised_price                    0
    discount_pct                        0
    vendor                              0
    global_popularity                   0
    count_adults                        0
    count_children                      0
    count_babies                        0
    count_pets                          0
    people_ex_baby                      0
    days_since_purchase_variant_id      0
    avg_days_to_buy_variant_id          0
    std_days_to_buy_variant_id          0
    days_since_purchase_product_type    0
    avg_days_to_buy_product_type        0
    std_days_to_buy_product_type        0
    dtype: int64




```python
# Verificar duplicados
duplicados = df.duplicated().sum()
print(f"Número de filas duplicadas: {duplicados}")


```

    Número de filas duplicadas: 0



```python
# Si hay duplicados, podemos eliminarlos
df = df.drop_duplicates()
```


```python
import seaborn as sns
import matplotlib.pyplot as plt

# Separar las variables categóricas y numéricas
variables_categoricas = ['product_type', 'vendor']
variables_numericas = [
    'variant_id', 'order_id', 'user_id', 'user_order_seq', 'outcome', 'ordered_before', 'abandoned_before', 
    'active_snoozed', 'set_as_regular', 'normalised_price', 'discount_pct', 'global_popularity', 
    'count_adults', 'count_children', 'count_babies', 'count_pets', 'people_ex_baby',
    'days_since_purchase_variant_id', 'avg_days_to_buy_variant_id', 'std_days_to_buy_variant_id',
    'days_since_purchase_product_type', 'avg_days_to_buy_product_type', 'std_days_to_buy_product_type'
]

# Crear gráficos para las variables categóricas
for var in variables_categoricas:
    plt.figure(figsize=(50, 10))
    sns.countplot(x=var, data=df)
    plt.title(f'Distribución de la variable categórica: {var}')
    plt.xticks(rotation=45)  # Rotar etiquetas si es necesario
    plt.show()

# Crear gráficos para las variables numéricas
for var in variables_numericas:
    plt.figure(figsize=(12, 10))
    df[var].hist(bins=30)
    plt.title(f'Distribución de la variable numérica: {var}')
    plt.show()

```


    
![png](EDA_2_files/EDA_2_6_0.png)
    



    
![png](EDA_2_files/EDA_2_6_1.png)
    



    
![png](EDA_2_files/EDA_2_6_2.png)
    



    
![png](EDA_2_files/EDA_2_6_3.png)
    



    
![png](EDA_2_files/EDA_2_6_4.png)
    



    
![png](EDA_2_files/EDA_2_6_5.png)
    



    
![png](EDA_2_files/EDA_2_6_6.png)
    



    
![png](EDA_2_files/EDA_2_6_7.png)
    



    
![png](EDA_2_files/EDA_2_6_8.png)
    



    
![png](EDA_2_files/EDA_2_6_9.png)
    



    
![png](EDA_2_files/EDA_2_6_10.png)
    



    
![png](EDA_2_files/EDA_2_6_11.png)
    



    
![png](EDA_2_files/EDA_2_6_12.png)
    



    
![png](EDA_2_files/EDA_2_6_13.png)
    



    
![png](EDA_2_files/EDA_2_6_14.png)
    



    
![png](EDA_2_files/EDA_2_6_15.png)
    



    
![png](EDA_2_files/EDA_2_6_16.png)
    



    
![png](EDA_2_files/EDA_2_6_17.png)
    



    
![png](EDA_2_files/EDA_2_6_18.png)
    



    
![png](EDA_2_files/EDA_2_6_19.png)
    



    
![png](EDA_2_files/EDA_2_6_20.png)
    



    
![png](EDA_2_files/EDA_2_6_21.png)
    



    
![png](EDA_2_files/EDA_2_6_22.png)
    



    
![png](EDA_2_files/EDA_2_6_23.png)
    



    
![png](EDA_2_files/EDA_2_6_24.png)
    



```python
# Convertir columnas de fecha a tipo datetime
df['created_at'] = pd.to_datetime(df['created_at'])
df['order_date'] = pd.to_datetime(df['order_date'])

# Convertir fechas a números (timestamp)
df['created_at_numeric'] = df['created_at'].view('int64') / 10**9  # En segundos
df['order_date_numeric'] = df['order_date'].view('int64') / 10**9  # En segundos

# Filtrar solo las columnas numéricas
numerical_df = df.select_dtypes(include=['float64', 'int64'])

# Añadir las columnas numéricas de las fechas convertidas
numerical_df['created_at_numeric'] = df['created_at_numeric']
numerical_df['order_date_numeric'] = df['order_date_numeric']

# Ajustar el tamaño de la figura
plt.figure(figsize=(14, 12))

# Matriz de correlación solo con variables numéricas (incluyendo fechas convertidas)
heatmap = sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 8})

# Ajustar el tamaño de las etiquetas de los ejes y rotarlas para mayor claridad
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=10)

plt.title('Matriz de Correlación (Solo Variables Numéricas y Fechas Convertidas)', fontsize=16)
plt.tight_layout()  # Ajustar la figura para que todo se vea bien
plt.show()



```


    
![png](EDA_2_files/EDA_2_7_0.png)
    



```python
# Ver valores únicos por columna
df.nunique()

```




    variant_id                           976
    product_type                          62
    order_id                            3446
    user_id                             1937
    created_at                          3446
    order_date                           149
    user_order_seq                        20
    outcome                                2
    ordered_before                         2
    abandoned_before                       2
    active_snoozed                         2
    set_as_regular                         2
    normalised_price                     127
    discount_pct                         526
    vendor                               264
    global_popularity                   5968
    count_adults                           5
    count_children                         4
    count_babies                           2
    count_pets                             5
    people_ex_baby                         5
    days_since_purchase_variant_id       142
    avg_days_to_buy_variant_id           122
    std_days_to_buy_variant_id           819
    days_since_purchase_product_type     141
    avg_days_to_buy_product_type          26
    std_days_to_buy_product_type          61
    created_at_numeric                  3446
    order_date_numeric                   149
    dtype: int64




```python
# Cambiar el tipo de datos si es necesario
# Por ejemplo, si una columna debería ser categórica
df['alguna_columna'] = df['alguna_columna'].astype('category')

```


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el DataFrame
df = pd.read_csv('/home/eduardo_1999/projects/zrive-ds/src/module_2/feature_frame.csv')

# Filtrar solo las columnas numéricas
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Configurar el tamaño de la figura
plt.figure(figsize=(20, 15))

# Crear un boxplot para cada variable numérica
for i, col in enumerate(numerical_columns):
    plt.subplot(5, 5, i + 1)  # Cambiar 5, 5 según cuántos subgráficos quieras en cada fila
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot de {col}')
    plt.xlabel(col)  # Etiquetar el eje x con el nombre de la variable

plt.tight_layout()  # Ajustar los espacios entre subgráficos
plt.show()


```


    
![png](EDA_2_files/EDA_2_10_0.png)
    



```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2880549 entries, 0 to 2880548
    Data columns (total 27 columns):
     #   Column                            Dtype  
    ---  ------                            -----  
     0   variant_id                        int64  
     1   product_type                      object 
     2   order_id                          int64  
     3   user_id                           int64  
     4   created_at                        object 
     5   order_date                        object 
     6   user_order_seq                    int64  
     7   outcome                           float64
     8   ordered_before                    float64
     9   abandoned_before                  float64
     10  active_snoozed                    float64
     11  set_as_regular                    float64
     12  normalised_price                  float64
     13  discount_pct                      float64
     14  vendor                            object 
     15  global_popularity                 float64
     16  count_adults                      float64
     17  count_children                    float64
     18  count_babies                      float64
     19  count_pets                        float64
     20  people_ex_baby                    float64
     21  days_since_purchase_variant_id    float64
     22  avg_days_to_buy_variant_id        float64
     23  std_days_to_buy_variant_id        float64
     24  days_since_purchase_product_type  float64
     25  avg_days_to_buy_product_type      float64
     26  std_days_to_buy_product_type      float64
    dtypes: float64(19), int64(4), object(4)
    memory usage: 593.4+ MB


**Analisis**_**de**_**datos**




##Hipótesis 1: El porcentaje de descuento (discount_pct) está positivamente relacionado con la probabilidad de compra (outcome).


```python
from scipy import stats

# Dividir los datos en dos grupos: compraron (outcome == 1) y no compraron (outcome == 0)
compraron = df[df['outcome'] == 1]['discount_pct'].dropna()
no_compraron = df[df['outcome'] == 0]['discount_pct'].dropna()

# Calcular medias
print("Media de descuento para los que compraron:", compraron.mean())
print("Media de descuento para los que no compraron:", no_compraron.mean())

# Test de comparación de medias (t-test)
t_stat, p_value = stats.ttest_ind(compraron, no_compraron)
print("t-statistic:", t_stat)
print("p-value:", p_value)


```

    Media de descuento para los que compraron: 0.188725209902251
    Media de descuento para los que no compraron: 0.18624581397472192
    t-statistic: 2.3229528564193753
    p-value: 0.02018175444663276


##2. Hipótesis: Los usuarios que han ordenado el producto antes (ordered_before) tienen mayor probabilidad de comprar.


```python
# Tabla de contingencia
contingency_table = pd.crosstab(df['ordered_before'], df['outcome'])

# Test de chi-cuadrado
chi2_stat, p_val, dof, ex = stats.chi2_contingency(contingency_table)

print("Tabla de contingencia:\n", contingency_table)
print("Chi-cuadrado:", chi2_stat)
print("p-valor:", p_val)

```

    Tabla de contingencia:
     outcome             0.0    1.0
    ordered_before                
    0.0             2796471  23187
    1.0               50846  10045
    Chi-cuadrado: 128400.13839567118
    p-valor: 0.0


##3. Hipótesis: La probabilidad de compra disminuye a medida que aumenta el precio normalizado (normalised_price).



```python
# Dividir los datos en dos grupos: compraron y no compraron
compraron = df[df['outcome'] == 1]['normalised_price'].dropna()
no_compraron = df[df['outcome'] == 0]['normalised_price'].dropna()

# Calcular medias
print("Media de precio para los que compraron:", compraron.mean())
print("Media de precio para los que no compraron:", no_compraron.mean())

# Test de comparación de medias (t-test)
t_stat, p_value = stats.ttest_ind(compraron, no_compraron)
print("t-statistic:", t_stat)
print("p-value:", p_value)

```

    Media de precio para los que compraron: 0.10437215717510694
    Media de precio para los que no compraron: 0.12754821851492537
    t-statistic: -33.1231743399502
    p-value: 1.5307890629527577e-240


##4. Hipótesis: Los productos con mayor popularidad global (global_popularity) tienen una mayor probabilidad de compra.



```python
compraron = df[df['outcome'] == 1]['global_popularity'].dropna()
no_compraron = df[df['outcome'] == 0]['global_popularity'].dropna()

print("Media de popularidad global para los que compraron:", compraron.mean())
print("Media de popularidad global para los que no compraron:", no_compraron.mean())

t_stat, p_value = stats.ttest_ind(compraron, no_compraron)
print("t-statistic:", t_stat)
print("p-value:", p_value)

```

    Media de popularidad global para los que compraron: 0.035742642768109754
    Media de popularidad global para los que no compraron: 0.010410773889688035
    t-statistic: 279.7385141231203
    p-value: 0.0


##5. Hipótesis: A mayor número de adultos en el hogar (count_adults), mayor es la probabilidad de compra.



```python
compraron = df[df['outcome'] == 1]['count_adults'].dropna()
no_compraron = df[df['outcome'] == 0]['count_adults'].dropna()

print("Media de adultos para los que compraron:", compraron.mean())
print("Media de adultos para los que no compraron:", no_compraron.mean())

t_stat, p_value = stats.ttest_ind(compraron, no_compraron)
print("t-statistic:", t_stat)
print("p-value:", p_value)

```

    Media de adultos para los que compraron: 2.026510592200289
    Media de adultos para los que no compraron: 2.0175235142416525
    t-statistic: 7.760444403225526
    p-value: 8.465982216873788e-15


##6. Hipótesis: Cuanto más reciente haya sido la última compra del mismo producto, mayor será la probabilidad de volver a comprarlo.



```python
compraron = df[df['outcome'] == 1]['days_since_purchase_variant_id'].dropna()
no_compraron = df[df['outcome'] == 0]['days_since_purchase_variant_id'].dropna()

print("Días promedio desde la última compra para los que compraron:", compraron.mean())
print("Días promedio desde la última compra para los que no compraron:", no_compraron.mean())

t_stat, p_value = stats.ttest_ind(compraron, no_compraron)
print("t-statistic:", t_stat)
print("p-value:", p_value)

```

    Días promedio desde la última compra para los que compraron: 33.912463890226284
    Días promedio desde la última compra para los que no compraron: 33.1204727116791
    t-statistic: 38.73025110788558
    p-value: 0.0


##7 Hipótesis: Los usuarios que han abandonado carritos previamente (abandoned_before) tienen menos probabilidades de comprar un producto.



```python
contingency_table = pd.crosstab(df['abandoned_before'], df['outcome'])

chi2_stat, p_val, dof, ex = stats.chi2_contingency(contingency_table)

print("Tabla de contingencia:\n", contingency_table)
print("Chi-cuadrado:", chi2_stat)
print("p-valor:", p_val)

```

    Tabla de contingencia:
     outcome               0.0    1.0
    abandoned_before                
    0.0               2846822  31972
    1.0                   495   1260
    Chi-cuadrado: 76783.13259360888
    p-valor: 0.0



```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo CSV desde la ruta indicada
file_path = '/home/eduardo_1999/projects/zrive-ds/src/module_2/feature_frame.csv'

df = pd.read_csv(file_path)

info_cols = ["variant_id", "order_id", "user_id", "created_at", "order_date"]
label_col = ["outcome"]
feature_cols = [col for col in df.columns if col not in info_cols + [label_col]]

categorical_cols =["product_type", "vendor"]
binary_cols = ["ordered_before","abandoned_before", "active_snoozed", "set_as_regular"]
numerical_cols = [col for col in feature_cols if col not in categorical_cols + binary_cols]


```


```python
df[label_col].value_counts()
```




    outcome
    0.0        2847317
    1.0          33232
    Name: count, dtype: int64




```python
for col in binary_cols:
	print(f"Value counts {col}: {df[col].value_counts().to_dict()}")
	print(
		f"Meanoutcome by {col} value: {df.groupby(col)['outcome'].mean().to_dict()}"
	)
	print(" --- ")
```

    Value counts ordered_before: {0.0: 2819658, 1.0: 60891}
    Meanoutcome by ordered_before value: {0.0: 0.008223337723936732, 1.0: 0.1649669080816541}
     --- 
    Value counts abandoned_before: {0.0: 2878794, 1.0: 1755}
    Meanoutcome by abandoned_before value: {0.0: 0.011106039542947498, 1.0: 0.717948717948718}
     --- 
    Value counts active_snoozed: {0.0: 2873952, 1.0: 6597}
    Meanoutcome by active_snoozed value: {0.0: 0.011302554809544488, 1.0: 0.1135364559648325}
     --- 
    Value counts set_as_regular: {0.0: 2870093, 1.0: 10456}
    Meanoutcome by set_as_regular value: {0.0: 0.010668992259135854, 1.0: 0.24971308339709258}
     --- 



```python
import numpy as np

cols = 3
rows = int(np.ceil(len(numerical_cols) / cols))
fig, ax = plt.subplots(rows, cols, figsize=(20, 5 * rows))
ax = ax.flatten()

for i, col in enumerate(numerical_cols):
	sns.kdeplot(df.loc[lambda x: x.outcome == 0, col], label="0", ax=ax[i])
	sns.kdeplot(df.loc[lambda x: x.outcome == 1, col], label="1", ax=ax[i])
	ax[i].set_title(col)

ax[0].legend()

plt.tight_layout
```

    /tmp/ipykernel_3094/1657019205.py:9: UserWarning: Dataset has 0 variance; skipping density estimate. Pass `warn_singular=False` to disable this warning.
      sns.kdeplot(df.loc[lambda x: x.outcome == 0, col], label="0", ax=ax[i])
    /tmp/ipykernel_3094/1657019205.py:10: UserWarning: Dataset has 0 variance; skipping density estimate. Pass `warn_singular=False` to disable this warning.
      sns.kdeplot(df.loc[lambda x: x.outcome == 1, col], label="1", ax=ax[i])





    <function matplotlib.pyplot.tight_layout(*, pad=1.08, h_pad=None, w_pad=None, rect=None)>




    
![png](EDA_2_files/EDA_2_30_2.png)
    



```python

```
