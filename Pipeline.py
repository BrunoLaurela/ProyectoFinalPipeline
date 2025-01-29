from ClasificadoresEnsamblados  import ClassifGSSC, ClassifYASA
from DetectoresEnsamblados import SpindleDetect,RemDetect,DetectorSW,Periodograma_Welch_por_segmento
import mne
import numpy as np
import pandas as pd
import yasa


#####################
# Autor: Bruno Laurela Luenlli
# Fecha: Ultima version : 29 de enero de 2025
# Descripción: Función para clasificar las etapas del sueño usando clasificadores 'YASA' y 'GSSC'
######################

def evaluacion_stim_channel(raw, channel_name, epoch_duration=30):
    """
    Evalúa los eventos de un canal específico en un conjunto de datos EEG.

    Esta función toma un objeto de datos EEG de MNE, selecciona un canal específico, 
    y evalúa los eventos en dicho canal para determinar en qué épocas (segmentos de tiempo)
    ocurrieron. Las épocas se determinan dividiendo el tiempo total de la señal por 
    una duración específica de época.

    Args:
        raw (mne.io.Raw): Objeto Raw de MNE que contiene los datos EEG.
        channel_name (str): Nombre del canal que se utilizará para la evaluación de eventos.
        epoch_duration (int, opcional): Duración de cada época en segundos. Por defecto es 30 segundos.

    Returns:
        pd.Series: Serie de pandas con la reevaluación de cada época. Cada índice representa una época,
                   y los valores representan el tiempo en segundos en el que ocurrió un evento dentro de esa época.
    """
    # Crear una copia del objeto raw y seleccionar el canal de interés
    raw_copy = raw.copy()
    raw_copy.pick_channels([channel_name])
    
    # Crear una serie de ceros con longitud igual al número de épocas
    Reevaluacion = np.zeros_like(np.arange(int(raw.n_times/raw.info['sfreq'])/epoch_duration))
    index = list(range(1, len(Reevaluacion)+1))
    Reevaluacion_pd = pd.Series(Reevaluacion, index=index)
    
    # Extraer eventos del canal de eventos
    events = mne.find_events(raw_copy)
    
    # Obtener las épocas en las que ocurrieron los eventos
    arrays = (events[:, 0] / raw.info['sfreq'] / epoch_duration).astype(int)
    muestras = (events[:, 0] / raw.info['sfreq']).astype(int)
    
    # Rellenar Reevaluacion_pd con las muestras ocurridas en cada época
    for id, muestra in zip(arrays, muestras):
        Reevaluacion_pd.loc[id] = muestra
    
    return Reevaluacion_pd

   
def AASM_RulesFINAL(raw, metadata, result):
   
    """
    Aplica reglas de la AASM para la clasificación de etapas del sueño basadas en diferentes características
    extraídas de las señales EEG y EMG.
    
    Parámetros:
    raw : mne.io.Raw
        Objeto con los datos EEG y EMG sin procesar.
    metadata : dict
        Diccionario con información sobre los canales y su distribución anatómica.
    result : dict
        Diccionario con los resultados del clasificador GSSC y pesos asociados.
    
    Retorna:
    Nueva_anotacion : list
        Lista con las nuevas anotaciones basadas en las reglas aplicadas.
    """
    
    Nueva_anotacion = []

    eventosSpindle = SpindleDetect(raw,metadata)
    eventosSW = DetectorSW(raw,metadata)
    DeteccionRem = RemDetect(raw,metadata)
    periodograma= Periodograma_Welch_por_segmento(raw,metadata)
    band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
    


    percentilSuperior_frW = []
    percentilSuperior_prW = []
    percentilSuperior_ocW = []
    percentilSuperior_crW = []

    percentilInferior_frW = []
    percentilInferior_prW = []
    percentilInferior_ocW = []
    percentilInferior_crW = []

    percentilSuperior_frN2 = []
    percentilSuperior_prN2 = []
    percentilSuperior_ocN2 = []
    percentilSuperior_crN2 = []

    percentilInferior_frN2 = []
    percentilInferior_prN2 = []
    percentilInferior_ocN2 = []
    percentilInferior_crN2 = []


    percentilSuperior_frN2Delta = []
    percentilSuperior_prN2Delta = []
    percentilSuperior_ocN2Delta = []
    percentilSuperior_crN2Delta = []

    percentilInferior_frN2Delta = []
    percentilInferior_prN2Delta = []
    percentilInferior_ocN2Delta = []
    percentilInferior_crN2Delta = []

    prediccion =  result['GSSC'][1]
    ############################### OBTENGO EL RMS #############################
    data = raw.get_data(metadata['channels']['emg'], units="uV")    ############## Elijo solo un canal de EEG  de todos los que tengo , no hay mucha diferencia entre canales de EEG #######################
    sf = raw.info['sfreq']
    # divido mi data en ventanas de 30 segundos
    _, data = yasa.sliding_window(data, sf, window=30)
    rms_values = np.sqrt(np.mean(data**2, axis=2))
    rms_list = rms_values.flatten()

    min_length = min(len(result['GSSC'][1]), len(rms_list))
    prediccion = result['GSSC'][1][:min_length]
    rms_list = rms_list[:min_length]

    # Identifica las épocas según la predicción
    EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]  
    EpocasW = [key for key, epoca in enumerate(prediccion) if epoca == 0]  
    EpocasN3 = [key for key, epoca in enumerate(prediccion) if epoca == 3]  
    EpocasR = [key for key, epoca in enumerate(prediccion) if epoca == 4]  
    # Filtra los valores RMS según las épocas
    RMS_N2 = rms_list[EpocasN2]
    RMS_N3 = rms_list[EpocasN3]
    RMS_W = rms_list[EpocasW]
    RMS_R = rms_list[EpocasR]
    # Cálculo de percentiles de RMS
    N2q90rmsEEG = np.nanpercentile(RMS_N2, 90)
    N3q90rmsEEG = np.nanpercentile(RMS_N3, 90)
    Rq90rmsEEG = np.nanpercentile(RMS_R, 90)
    Wq90rmsEEG = np.nanpercentile(RMS_W, 90)
    Wq5rmsEEG = np.nanpercentile(RMS_W, 5)
    # Cálculo de percentiles de pesos
    PesosN2Q10 = np.nanpercentile(result['GSSC'][0]['N2'][EpocasN2], 10) # pongo 10 porque es la menor cantidad de epcoas con las que suele confundirse con N1
    PesosWQ5= np.nanpercentile(result['GSSC'][0]['W'][EpocasW],5)

        
    # Iteración sobre las regiones cerebrales y cálculos de percentiles para cada banda de frecuencia

    for region in ['occipital', 'frontal', 'central', 'parietal']:
        if metadata['channels']['eeg'][region]:
            canales = metadata['channels']['eeg'][region]
            
            for ch in canales:
                ### Delta
                min_length = min(len(prediccion), len(periodograma.loc['Delta', ch]))
                peri_Delta = periodograma.loc['Delta', ch][:min_length]
                ## Para N2
                EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]   
                filas_seleccionadasDeltaN2 = peri_Delta.loc[EpocasN2]
                ### Alpha
                min_length = min(len(prediccion), len(periodograma.loc['Alpha', ch]))
                peri_alpha = periodograma.loc['Alpha', ch][:min_length]
                ## Para N2
                EpocasN2 = [key for key, epoca in enumerate(prediccion) if epoca == 2]   
                filas_seleccionadasAlphaN2 = peri_alpha.loc[EpocasN2]
                ### Tetha 
                min_length = min(len(prediccion), len(periodograma.loc['Theta', ch]))
                peri_Theta = periodograma.loc['Theta', ch][:min_length]
                ## Para W
                EpocasW= [key for key, epoca in enumerate(prediccion) if epoca == 0]   
                filas_seleccionadasThetaW = peri_Theta.loc[EpocasW]
                if region == 'occipital':
                    percentilSuperior_ocW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 95))
                    percentilInferior_ocW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentilSuperior_ocN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 95))
                    percentilInferior_ocN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentilSuperior_ocN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 95))
                    percentilInferior_ocN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'frontal':
                    percentilSuperior_frW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 95))
                    percentilInferior_frW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentilSuperior_frN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 95))
                    percentilInferior_frN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentilSuperior_frN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 95))
                    percentilInferior_frN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'central':
                    percentilSuperior_crW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 95))
                    percentilInferior_crW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentilSuperior_crN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 95))
                    percentilInferior_crN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentilSuperior_crN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 95))
                    percentilInferior_crN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'parietal':
                    percentilSuperior_prW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 95))
                    percentilInferior_prW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentilSuperior_prN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 95))
                    percentilInferior_prN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentilSuperior_prN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 95))
                    percentilInferior_prN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
            
    QSuperior_fr_MaxW = max(percentilSuperior_frW) if percentilSuperior_frW else 0
    QSuperior_pr_MaxW = max(percentilSuperior_prW) if percentilSuperior_prW else 0
    QSuperior_oc_MaxW = max(percentilSuperior_ocW) if percentilSuperior_ocW else 0
    QSuperior_cr_MaxW = max(percentilSuperior_crW) if percentilSuperior_crW else 0

    QInferior_fr_MaxW = max(percentilInferior_frW) if percentilInferior_frW else 0
    QInferior_pr_MaxW = max(percentilInferior_prW) if percentilInferior_prW else 0
    QInferior_oc_MaxW = max(percentilInferior_ocW) if percentilInferior_ocW else 0
    QInferior_cr_MaxW = max(percentilInferior_crW) if percentilInferior_crW else 0

    QSuperior_fr_MaxN2_alpha = max(percentilSuperior_frN2) if percentilSuperior_frN2 else 0
    QSuperior_pr_MaxN2_alpha = max(percentilSuperior_prN2) if percentilSuperior_prN2 else 0
    QSuperior_oc_MaxN2_alpha = max(percentilSuperior_ocN2) if percentilSuperior_ocN2 else 0
    QSuperior_cr_MaxN2_alpha = max(percentilSuperior_crN2) if percentilSuperior_crN2 else 0

    QInferior_fr_MaxN2_alpha = max(percentilInferior_frN2) if percentilInferior_frN2 else 0
    QInferior_pr_MaxN2_alpha = max(percentilInferior_prN2) if percentilInferior_prN2 else 0
    QInferior_oc_MaxN2_alpha = max(percentilInferior_ocN2) if percentilInferior_ocN2 else 0
    QInferior_cr_MaxN2_alpha = max(percentilInferior_crN2) if percentilInferior_crN2 else 0


    QSuperior_fr_MaxN2_Delta = max(percentilSuperior_frN2Delta) if percentilSuperior_frN2Delta else 0
    QSuperior_pr_MaxN2_Delta = max(percentilSuperior_prN2Delta) if percentilSuperior_prN2Delta else 0
    QSuperior_oc_MaxN2_Delta = max(percentilSuperior_ocN2Delta) if percentilSuperior_ocN2Delta else 0
    QSuperior_cr_MaxN2_Delta = max(percentilSuperior_crN2Delta) if percentilSuperior_crN2Delta else 0

    QInferior_fr_MaxN2_Delta = max(percentilInferior_frN2Delta) if percentilInferior_frN2Delta else 0
    QInferior_pr_MaxN2_Delta = max(percentilInferior_prN2Delta) if percentilInferior_prN2Delta else 0
    QInferior_oc_MaxN2_Delta = max(percentilInferior_ocN2Delta) if percentilInferior_ocN2Delta else 0
    QInferior_cr_MaxN2_Delta = max(percentilInferior_crN2Delta) if percentilInferior_crN2Delta else 0




    # Dado que GSSC corta el hinograma en caso que las epcoas no coincidan con segmentos exactos de 30 se hace lo siguiente
    # para manetener el mismo numero de anotaciones tanto en las predicciones como en  la lista que tiene las epoca a reevaluar 
    Numero_de_epocas = int((raw.n_times/raw.info['sfreq'])/30)
    Reevaluacion_pd = [i * 30 for i in range(Numero_de_epocas)]   
    reevaluacion = Reevaluacion_pd
    predicciones =  result['GSSC'][1]
    if len(reevaluacion) > len( result['GSSC'][1]):
        # Recortar reevaluacion al tamaño de predicciones eliminando el último elemento (generalmente lso algoritmos de clasificacion recortan la señal en el ultimo extremo)
        Reevaluacion_pd = reevaluacion[:len(predicciones)]

    for epoca, (deteccion, etapa) in enumerate(zip(Reevaluacion_pd,  result['GSSC'][1])):
        print('EPOCA', epoca)

        band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
        if metadata['channels']['eeg']['occipital']:
            canales = metadata['channels']['eeg']['occipital']
            Porcentaje_de_densidad_alpha_oc = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_oc = max(periodograma.loc['Delta', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Theta_oc = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_oc,  Porcentaje_de_densidad_Delta_oc, Porcentaje_de_densidad_Theta_oc = 0,0,0
        if metadata['channels']['eeg']['frontal']:
            canales = metadata['channels']['eeg']['frontal']
            Porcentaje_de_densidad_alpha_fr =max( periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_fr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_fr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_fr,  Porcentaje_de_densidad_Delta_fr, Porcentaje_de_densidad_Theta_fr = 0,0,0
        if metadata['channels']['eeg']['central']:
            canales = metadata['channels']['eeg']['central']
            Porcentaje_de_densidad_alpha_cr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_cr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_cr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_cr,  Porcentaje_de_densidad_Delta_cr, Porcentaje_de_densidad_Theta_cr = 0,0,0
        if metadata['channels']['eeg']['parietal']:
            canales = metadata['channels']['eeg']['parietal']
            Porcentaje_de_densidad_alpha_pr = max(periodograma.loc['Alpha', canales].loc[epoca])    # podria evaluar la transicion de la etapa del despertar al sueño 
            Porcentaje_de_densidad_Delta_pr = max(periodograma.loc['Delta', canales].loc[epoca])
            Porcentaje_de_densidad_Theta_pr = max(periodograma.loc['Theta', canales].loc[epoca])
        else :  Porcentaje_de_densidad_alpha_pr,  Porcentaje_de_densidad_Delta_pr, Porcentaje_de_densidad_Theta_pr = 0,0,0
        ################################

        ######################### obtengo el valor de RMS de Q90 para la epoca de evaluacion #######################################
        
        RMSenEpocaActual = rms_list[epoca]
        ############################### PESO EPOCA ACTUAL ##################################
        peso_actual = result['GSSC'][0]['W'][epoca]
        peso_actualN2 = result['GSSC'][0]['N2'][epoca]
        ############################################################################################################################
        start_time = deteccion  # las detecciones siempre vana  estar redondeadas en 30 segundos, proque se marcan al incio de cada epoca
        first_half_period = deteccion + 15
        previous_half_period = deteccion-15
        end_time = start_time + 30  
        # Verificar que los tiempos estén dentro del rango
        if start_time >= eventosSW['Start'].min() and end_time <= eventosSW['Start'].max():
            # Aquí se evalúan las condiciones si estamos dentro del rango de eventos
            condicion = (
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period) |
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time)
            )
            eventos_en_rango_sw = eventosSW['Start'][condicion]
            duracion__en_rango_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & 
                (eventosSW['Start'] < end_time)
            ]
        else:
            # Manejo de error o advertencia
            print("La detección cae fuera de los límites de la lista de eventos")
            eventos_en_rango_sw = []  # O algún valor predeterminado
            duracion__en_rango_sw = []
                # Verificar los límites para Spindles
        if start_time >= eventosSpindle.min() :
            eventos_en_rango_spindle = eventosSpindle[
                ((eventosSpindle >= start_time) & (eventosSpindle < first_half_period)) |
                ((eventosSpindle >= previous_half_period) & (eventosSpindle < start_time))
            ]
        else:
            print("Los tiempos de detección para Spindles están fuera de los límites de la lista")
            eventos_en_rango_spindle = []  # O algún valor predeterminado
        
        if DeteccionRem.size == 1:
            min_rem = DeteccionRem.min().item()
            max_rem = DeteccionRem.max().item()
        else:
            min_rem = DeteccionRem.min()
            max_rem = DeteccionRem.max()

        # Solo proceder si min_rem y max_rem tienen el formato adecuado
        if (start_time >= min_rem).any and (end_time <= max_rem).any:
            eventos_en_rango_REM = DeteccionRem[ (DeteccionRem >= start_time) & (DeteccionRem < end_time)]
        else:
            print("Los tiempos de detección para REM están fuera de los límites de la lista")
            eventos_en_rango_REM = [] 
    
        start_time = deteccion  - 30 # epoca anterior
        first_half_period = deteccion + 15
        previous_half_period = deteccion-15
        end_time = start_time + 30  
        # Verificar los límites para eventos Spindle de la época anterior
        if previous_half_period >= eventosSpindle.min() and start_time <= eventosSpindle.max():
            eventos_en_rango_EpocaAnterior_spindle = eventosSpindle[
                ((eventosSpindle >= previous_half_period) & (eventosSpindle < start_time)) |
                ((eventosSpindle >= start_time) & (eventosSpindle < first_half_period))
            ]
        else:
            eventos_en_rango_EpocaAnterior_spindle = []  # o un valor predeterminado

        # Verificar los límites para eventos SW en la época anterior
        if previous_half_period >= eventosSW['Start'].min() and start_time <= eventosSW['Start'].max():
            condicion2 = (
                (eventosSW['Start'] >= previous_half_period) & (eventosSW['Start'] < start_time) |
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < first_half_period)
            )
            eventos_en_rango_EpocaAnterior_sw = eventosSW['Start'][condicion2]
            duracion__en_rango_EpocaAnterior_sw = eventosSW['Duration'][
                (eventosSW['Start'] >= start_time) & (eventosSW['Start'] < end_time)
            ]
        else:
            eventos_en_rango_EpocaAnterior_sw = []
            duracion__en_rango_EpocaAnterior_sw = []
    
            
        # EVALUO N1

        if etapa == 2:
            # Aca me fijo si la etapa evaluada como N2 realmente corresponde a N2
            if  len(eventos_en_rango_spindle) != 0 or len(eventos_en_rango_sw) != 0 :
                Nueva_anotacion.append(2)
            elif len(eventos_en_rango_EpocaAnterior_spindle) != 0 or len(eventos_en_rango_EpocaAnterior_sw) != 0:   
                Nueva_anotacion.append(2)
            elif Porcentaje_de_densidad_alpha_cr > QSuperior_cr_MaxN2_alpha or  Porcentaje_de_densidad_alpha_fr > QSuperior_fr_MaxN2_alpha or Porcentaje_de_densidad_alpha_pr > QSuperior_pr_MaxN2_alpha or Porcentaje_de_densidad_alpha_oc > QSuperior_oc_MaxN2_alpha:
                print('ES N1')
                Nueva_anotacion.append(1)
            elif peso_actualN2 < PesosN2Q10 :
                Nueva_anotacion.append(1)
            elif np.sum(duracion__en_rango_sw) > 0.2*30:
                Nueva_anotacion.append(3)
            elif len(eventos_en_rango_REM) != 0:
                Nueva_anotacion.append(4)    

            else :
                Nueva_anotacion.append(2)
                print("No se cumplieron las condiciones específicas para N1, queda como N2")
    
        elif etapa == 0:
            if Porcentaje_de_densidad_Theta_cr > QSuperior_cr_MaxW or Porcentaje_de_densidad_Theta_fr > QSuperior_fr_MaxW or Porcentaje_de_densidad_Theta_pr > QSuperior_pr_MaxW or Porcentaje_de_densidad_Theta_oc > QSuperior_oc_MaxW:
                Nueva_anotacion.append(1)
                print('ES N1')
            elif (RMSenEpocaActual < Wq5rmsEEG )  and ( peso_actual < PesosWQ5 ) :
                Nueva_anotacion.append(1)
                print('ES N1')
            else:
                print("No se cumplieron las condiciones específicas para W, queda como W")
                Nueva_anotacion.append(0)
        

        else:
            # Este else final se ejecuta si ninguna de las condiciones anteriores es verdadera
            Nueva_anotacion.append(result['GSSC'][1][epoca])
    
    return Nueva_anotacion
   
def classify_file(file_data, metadata, classifiers):
    """
    Clasifica un archivo de datos de sueño utilizando los clasificadores especificados.

    Esta función recibe datos de sueño, metadatos y una lista de clasificadores, 
    y devuelve un diccionario donde cada nombre de clasificador se asocia con su 
    resultado de clasificación correspondiente.

    Argumentos:
        file_data (dict o np.array): Datos de sueño a clasificar. Los datos pueden estar en formato de diccionario o en un arreglo numpy.
        metadata (dict): Metadatos adicionales relacionados con el archivo de datos de sueño. Esto puede incluir información como 
                         el sujeto, configuraciones de grabación o marcas de tiempo.
        classifiers (list): Una lista de nombres de clasificadores a aplicar. Las entradas válidas son 'YASA' para el clasificador YASA y 
                            'GSSC' para el clasificador GSSC. Se pueden proporcionar uno o más clasificadores en la lista.

    Retorna:
        dict: Un diccionario donde las claves son los nombres de los clasificadores (por ejemplo, 'YASA', 'GSSC') y los valores son los 
              resultados de clasificación correspondientes. El formato de cada resultado depende de la salida del clasificador correspondiente.
    """
   
    results = {}

    for clasiff   in classifiers :
        if clasiff == 'YASA':
            result = ClassifYASA(file_data, metadata)
        if clasiff == 'GSSC' :
            result = ClassifGSSC(file_data, metadata)
        results[clasiff] = result
    return results
def sleep_stage_classification_file(file_data, metadata, classifiers = None):
    """
    Clasifica las etapas del sueño en un archivo de datos de polisomnografía usando clasificadores seleccionados.
    
    Args:
        file_data (mne.raw): Archivo `raw` de MNE que contiene la señal de polisomnografía del estudio.
        metadata (dict): Diccionario con las variables necesarias para realizar la clasificación. 
                         Debe tener el siguiente formato:
                         dict = {
                             'channels': {
                                 'eog': [],
                                 'eeg': {'frontal': [], 'central': [], 'parietal': []},
                                 'emg': []
                             }
                         }
        classifiers (list, opcional): Lista con los nombres de los clasificadores a usar. Si se proporciona, debe contener al menos dos clasificadores. 
                                      Por defecto, los clasificadores son ['YASA', 'GSSC'].

    Returns:
        array: Retorna un array con los valores de la clasificación final de las etapas del sueño.
    """
    if  classifiers == None :
        classifiers = ['YASA', 'GSSC']
   
    results = classify_file(file_data, metadata, classifiers) 
  
    prediccion= AASM_RulesFINAL(file_data, metadata, results)
    return prediccion
