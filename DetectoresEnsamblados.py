import numpy as np
import pandas as pd
import seaborn as sns
import mne_bids 
import mne
import os 
import yasa
import pandas as pd
from KC_algorithm.model import score_KCs
from KC_algorithm.plotting import KC_from_probas
from scipy.signal import welch
import warnings
#####################
# Autor: Bruno Laurela Luenlli
# Fecha: Ultima version : 29 de enero de 2025
# Descripción: Función para clasificar las etapas del sueño usando clasificadores 'YASA' y 'GSSC'
######################

##############################################################
# El proposito de este modulo es generar funciones que para un evento en particular tengan todas las mismas salidas y entradas

# Las salidas de estas funciones peuden ser de dos tipos : Series de Pandas o Archivos raw con canales Stim indicando el evento
##############################################################

# detectar eventos spindle  tengo que seguir trabajando en este codigo

def SpindleDetect(raw, metadata): # parametros : definir ...
    """
    Detecta los husos de sueño en los datos de EEG utilizando el método de YASA.

    Esta función selecciona los canales EEG especificados en los metadatos y aplica la detección de husos de sueño.
    Si se detectan husos de sueño, la función devuelve las marcas de tiempo de inicio de los eventos de husos únicos.

    Argumentos:
        raw (mne.io.Raw): Datos de EEG en formato de objeto MNE Raw.
        metadata (dict): Metadatos relacionados con el archivo de datos de EEG, 
                         que incluyen los canales de EEG a utilizar para la detección de husos.

    Retorna:
        pandas.Series: Una serie con los tiempos de inicio de los eventos de husos de sueño detectados y filtrados.
    """
    # seleciono los canales que voy a evaluar spindle
    raw_copy = raw.copy()

    eeg_list = []
    for subkey in metadata['channels']['eeg']:
        eeg_list.extend(metadata['channels']['eeg'][subkey])
    raw_copy.pick( eeg_list )
        
    spindle = yasa.spindles_detect(raw_copy) # suiil

   

   # Verificar si no se encontraron husos de sueño por algun error
    if spindle is None :
        warnings.warn("No se encontraron husos de sueño en los datos.")
        eventos = pd.DataFrame()  # DataFrame vacío
    else:
        
        events = spindle.summary()
        
        events_sorted = events.sort_values('Start').reset_index(drop=True)

        # Eliminar eventos duplicados dentro de un margen de tiempo (por ejemplo, 100 milisegundos)
        margin = 0.5  # 500 milisegundos
        unique_events = [events_sorted.iloc[0]]  # Lista para guardar eventos únicos

        for i in range(1, len(events_sorted)):
            if events_sorted['Start'].iloc[i] - unique_events[-1]['Start'] > margin:
                unique_events.append(events_sorted.iloc[i])

        # Convertir la lista de eventos únicos de nuevo a un DataFrame
        unique_events_df = pd.DataFrame(unique_events)
        
        eventos = unique_events_df['Start']
 
    return eventos
   


def kComplexDetect(raw, canal, dict, Annotacion_con__duracion  = True) :
    
    """
    raw (mne.io.Raw): Datos en crudo con las anotaciones ya establecidas. 
                      Es importante que las anotaciones estén correctamente seteadas 
                      en el objeto `raw` y que la duración de cada evento sea de 30 segundos 
                      para cada anotación.

    dict (dict): Diccionario que contiene las etiquetas numéricas que deseas asignar a los eventos 
                 en base a las anotaciones.

    canal (str): Nombre del canal seleccionado para la detección de complejos K.

    Retorna:
        tuple: Una tupla con los siguientes elementos:
            - onsets_ (np.array): Los índices de las muestras donde se detectan los complejos K.
            - raw (mne.io.Raw): El objeto `raw` con el canal de estímulo añadido para los complejos K.
    """
    if Annotacion_con__duracion :
        events_train, _ = mne.events_from_annotations(
            raw, event_id= dict, chunk_duration= 30.0)
        hypno = pd.DataFrame()
        hypno['onset'] = events_train[:,0]
        hypno['dur'] = [30 for _ in events_train[:,1]]
        hypno['label'] = events_train[:,2]
    else :
        events_train, _ = mne.events_from_annotations(
            raw, event_id= dict)
        hypno = pd.DataFrame()
        hypno['onset'] = events_train[:,0]/raw.info['sfreq']
        hypno['dur'] = np.ones_like(events_train[:,0])*30
        hypno['label'] = events_train[:,2]

    # clasifico los k complex
    wanted_channel = canal #
    CZ = np.asarray(
        [raw[count, :][0]for count, k in enumerate(raw.info['ch_names']) if
            k == wanted_channel]).ravel()*-1

    Fs = raw.info['sfreq']

    peaks, stage_peaks, d, probas = score_KCs(CZ, Fs, hypno,sleep_stages=list(dict.values()))  # list(dict.values()) me permite evaluar posibles complejos K en cualquier etapa evaluada
   
    # me quedo con  aquellos k complex que tienen mas del 80% de probabilidad
    onsets_ = peaks[probas>0.8]  # indica el numero de muesstra donde se encuentra elcomplejo k
    
    # genero un canal stim y agrego los eventos
    pre_stim =np.zeros_like(np.array(np.arange(raw.n_times))) #[:,:][0][0]
    pre_stim[onsets_] = 1
    stim_chan = pre_stim.reshape(1,-1)

    mask_info = mne.create_info(ch_names=["STIMcomplexK"],
                                sfreq=raw.info["sfreq"],
                                ch_types=["stim"]
                            )
    raw_mask = mne.io.RawArray(data=stim_chan,
                            info=mask_info,
                            first_samp=raw.first_samp
                            )
    raw.add_channels([raw_mask], force_update_info=True)
    
    return  onsets_ , raw

# Revisar implementacion SW y REMdetect

def RemDetect(raw,metadata) :   # los datos debene estar en uV
    """ 
    Detecta los eventos de REM a partir de los datos EOG.

    Args:
        loc (type): Descripción de este parámetro (por ejemplo, ubicación de los datos o tipo de análisis que se realiza con él).
        roc (type): Descripción del parámetro relacionado con el análisis.
        sf (type): Frecuencia de muestreo de los datos (Hz).
        raw (mne.io.Raw): Datos crudos (de tipo `mne.io.Raw`) que contienen las señales EOG y otros canales necesarios para la detección de REM.

    Returns:
        events (pandas.Series): Una serie de pandas que contiene los inicios de los eventos REM detectados.
    """
    if len(metadata['channels']['eog']) == 2 :
        rem_ = yasa.rem_detect(raw.get_data(picks= metadata['channels']['eog'], units = 'uV')[0], raw.get_data(picks= metadata['channels']['eog'][1],units = 'uV'),raw.info['sfreq'], duration=(0.2, 0.6))

         # Get the detection dataframe
        if rem_ is None :
            eventos = pd.DataFrame()
        else :
            events = rem_.summary()
            eventos = events['Start']
    else :
        eventos= pd.DataFrame()
         
 
    return eventos #raw


def DetectorSW(raw,metadata):   # puedo usarlo tambien para detectar complejos K teniendo encuenta que son identicamente iguales a las ondas lentas pero aisladas en una etapa
    """
    Detecta los eventos de ondas lentas (SW) en los datos de EEG. Este detector también puede ser utilizado
    para detectar complejos K, ya que son similares a las ondas lentas pero aisladas en una etapa de sueño específica.

    Args:
        raw (mne.io.Raw): Datos EEG crudos (tipo `mne.io.Raw`) que contienen las señales de interés.
        metadata (dict): Diccionario con información adicional, como los canales EEG para el análisis.

    Returns:
        eventos (pandas.DataFrame): Un DataFrame con las columnas 'Start' y 'Duration', donde:
            - 'Start': Indica el inicio de cada evento detectado (en segundos).
            - 'Duration': Duración de cada evento detectado (en segundos).
    """
    raw_copy = raw.copy()
    eeg_list = []
    for subkey in metadata['channels']['eeg']:
        eeg_list.extend(metadata['channels']['eeg'][subkey])
    raw_copy.pick( eeg_list)
    
    sw = yasa.sw_detect(raw_copy)
    # Get the detection dataframe
    if sw is None :
        warnings.warn("No se encontraron husos de sueño en los datos.")
        eventos = pd.DataFrame()  # DataFrame vacío
    else:
        events = sw.summary()


        events_sorted = events.sort_values('Start').reset_index(drop=True)

        # Eliminar eventos duplicados dentro de un margen de tiempo (por ejemplo, 100 milisegundos)
        margin = 0.5  # 500 milisegundos
        unique_events = [events_sorted.iloc[0]]  # Lista para guardar eventos únicos

        for i in range(1, len(events_sorted)):
            if events_sorted['Start'].iloc[i] - unique_events[-1]['Start'] > margin:
                unique_events.append(events_sorted.iloc[i])

        # Convertir la lista de eventos únicos de nuevo a un DataFrame
        unique_events_df = pd.DataFrame(unique_events)
        eventos =unique_events_df[['Start','Duration']]

    return eventos
    


def Periodograma_Welch_por_segmento(raw, metadata) :
    """
    Esta función calcula la potencia espectral de las bandas de frecuencia para los canales seleccionados
    (EEG, EMG y EOG) de los datos crudos de EEG y devuelve un DataFrame con la potencia de cada banda en
    cada canal. La potencia se calcula utilizando el método de Welch para obtener la densidad espectral de
    potencia (PSD), seguida de la descomposición en bandas de frecuencia.

    Args:
        raw (mne.io.Raw): Datos crudos de EEG (tipo `mne.io.Raw`) que contienen las señales de interés.
        metadata (dict): Diccionario con la información de los canales (EEG, EMG, EOG) a analizar.

    Returns:
        result_df (pandas.DataFrame): Un DataFrame con la potencia de cada banda en cada canal.
            Las bandas se encuentran organizadas por tipo (Delta, Theta, Alpha, Sigma, Beta) en filas,
            y los canales (EEG, EMG, EOG) en columnas.
    """
    
    # Selecionon de canales  solo elijo 1 canal de EEG 1 de EMG  y los dos de EOG:
    regiones = ['frontal', 'central', 'parietal', 'occipital']
    Canal = []
    for region in regiones:
        if metadata['channels']['eeg'][region]:
            if len(metadata['channels']['eeg'][region]) >= 2:
                Canal.extend(metadata['channels']['eeg'][region])
            else:
                Canal.append(metadata['channels']['eeg'][region][0])

    # Agregar canal EMG si existe
    if metadata['channels']['emg']:
        Canal.append(metadata['channels']['emg'][0])

    # Agregar canales EOG si existen exactamente dos
    if 'eog' in metadata['channels'] and len(metadata['channels']['eog']) == 2:
        Canal.extend(metadata['channels']['eog'])


    # Create a 3-D array
    data = raw.get_data(Canal, units="uV")    ############## Elijo solo un canal de EEG  de todos los que tengo , no hay mucha diferencia entre canales de EEG #######################
    sf = raw.info['sfreq']

    # divido mi data en ventanas de 30 segundos
    _, data = yasa.sliding_window(data, sf, window=30)

    #print(raw.ch_names)
    #print(data.shape, sf)   

    # calculo la ventana para calcular con PSD la densidad espectrar de potencia a los canales seleccionados
    win = int(4 * sf)  # Window size is set to 4 seconds
    freqs, psd = welch(data, sf, nperseg=win, axis=-1) 

    #freqs.shape, psd.shape

    # separo en bandas frecuenciales lo calculado 
    bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), 
        (12, 16, 'Sigma'), (16, 30, 'Beta')]

    # Calculate the bandpower on 3-D PSD array 
    bandpower = yasa.bandpower_from_psd_ndarray(psd, freqs, bands)
    bandpower = np.round(bandpower, 3)

    band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']

    # Convertir cada banda de frecuencia en un DataFrame separado, de esta manera se presentan mas ordenados
    dfs = []
    for  i in range(bandpower.shape[0]):  # Iterar sobre las bandas de frecuencia
        df = pd.DataFrame(bandpower[i], columns=Canal)
        dfs.append(df)

    result_df = pd.concat(dfs, keys=[f'{banda}' for i, banda in zip(range(bandpower.shape[0]),band_names)], axis=0)

    
    return  result_df