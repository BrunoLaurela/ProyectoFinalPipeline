
ESTE ES EL CODIGO DE aasm QUE USE PARA ESTA PRUEBA. e esta prueba uso otro enfoque a las anteriores para evaluar las etapas, no necesito encontrar los puntos donde el algoritmos e equivoca
def ASSM_RulesDirectConReevaluacion(raw, metadata, result):
    # pregutnar primero N2 salgo
    # sino pregunto N3  salgo
    # sino es N3  fijarme  -> REM sino tiene rem puede ser 
    # fiajrme sino rem si tiene si tiene directamente movimeinto rapido de ojos preguntar si es WAKE con lo de alpha --> sino es esto peude ser rem o N1  , si vengo de vigilia N1 si la enterior es rem . rem
    
    Nueva_anotacion = []

    eventosSpindle = SpindleDetect(raw,metadata)
    eventosSW = DetectorSW(raw,metadata)
    #DeteccionRem = RemDetect(raw,metadata)
    periodograma= Periodograma_Welch_por_segmento(raw,metadata)
    band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
    


    percentil75_frW = []
    percentil75_prW = []
    percentil75_ocW = []
    percentil75_crW = []

    percentil25_frW = []
    percentil25_prW = []
    percentil25_ocW = []
    percentil25_crW = []

    percentil75_frN2 = []
    percentil75_prN2 = []
    percentil75_ocN2 = []
    percentil75_crN2 = []

    percentil25_frN2 = []
    percentil25_prN2 = []
    percentil25_ocN2 = []
    percentil25_crN2 = []


    percentil75_frN2Delta = []
    percentil75_prN2Delta = []
    percentil75_ocN2Delta = []
    percentil75_crN2Delta = []

    percentil25_frN2Delta = []
    percentil25_prN2Delta = []
    percentil25_ocN2Delta = []
    percentil25_crN2Delta = []
    prediccion =  result['GSSC'][1]

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
                    percentil75_ocW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 99))
                    percentil25_ocW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_ocN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 99))
                    percentil25_ocN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_ocN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 99))
                    percentil25_ocN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'frontal':
                    percentil75_frW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 99))
                    percentil25_frW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_frN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 99))
                    percentil25_frN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_frN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 99))
                    percentil25_frN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'central':
                    percentil75_crW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 99))
                    percentil25_crW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_crN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 99))
                    percentil25_crN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_crN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 99))
                    percentil25_crN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
                if region == 'parietal':
                    percentil75_prW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 99))
                    percentil25_prW.append(np.nanpercentile(filas_seleccionadasThetaW.values, 2))
                    percentil75_prN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 99))
                    percentil25_prN2.append(np.nanpercentile(filas_seleccionadasAlphaN2.values, 2))
                    percentil75_prN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 99))
                    percentil25_prN2Delta.append(np.nanpercentile(filas_seleccionadasDeltaN2.values, 2))
            
    Q75_fr_MaxW = max(percentil75_frW) if percentil75_frW else 0
    Q75_pr_MaxW = max(percentil75_prW) if percentil75_prW else 0
    Q75_oc_MaxW = max(percentil75_ocW) if percentil75_ocW else 0
    Q75_cr_MaxW = max(percentil75_crW) if percentil75_crW else 0

    Q25_fr_MaxW = max(percentil25_frW) if percentil25_frW else 0
    Q25_pr_MaxW = max(percentil25_prW) if percentil25_prW else 0
    Q25_oc_MaxW = max(percentil25_ocW) if percentil25_ocW else 0
    Q25_cr_MaxW = max(percentil25_crW) if percentil25_crW else 0

    Q75_fr_MaxN2_alpha = max(percentil75_frN2) if percentil75_frN2 else 0
    Q75_pr_MaxN2_alpha = max(percentil75_prN2) if percentil75_prN2 else 0
    Q75_oc_MaxN2_alpha = max(percentil75_ocN2) if percentil75_ocN2 else 0
    Q75_cr_MaxN2_alpha = max(percentil75_crN2) if percentil75_crN2 else 0

    Q25_fr_MaxN2_alpha = max(percentil25_frN2) if percentil25_frN2 else 0
    Q25_pr_MaxN2_alpha = max(percentil25_prN2) if percentil25_prN2 else 0
    Q25_oc_MaxN2_alpha = max(percentil25_ocN2) if percentil25_ocN2 else 0
    Q25_cr_MaxN2_alpha = max(percentil25_crN2) if percentil25_crN2 else 0


    Q75_fr_MaxN2_Delta = max(percentil75_frN2Delta) if percentil75_frN2Delta else 0
    Q75_pr_MaxN2_Delta = max(percentil75_prN2Delta) if percentil75_prN2Delta else 0
    Q75_oc_MaxN2_Delta = max(percentil75_ocN2Delta) if percentil75_ocN2Delta else 0
    Q75_cr_MaxN2_Delta = max(percentil75_crN2Delta) if percentil75_crN2Delta else 0

    Q25_fr_MaxN2_Delta = max(percentil25_frN2Delta) if percentil25_frN2Delta else 0
    Q25_pr_MaxN2_Delta = max(percentil25_prN2Delta) if percentil25_prN2Delta else 0
    Q25_oc_MaxN2_Delta = max(percentil25_ocN2Delta) if percentil25_ocN2Delta else 0
    Q25_cr_MaxN2_Delta = max(percentil25_crN2Delta) if percentil25_crN2Delta else 0




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
        """
        # Verificar los límites para REM
        if (start_time >= DeteccionRem.min().item()) and (end_time <= DeteccionRem.max().item()):

            eventos_en_rango_REM = DeteccionRem[
                (DeteccionRem >= start_time) & (DeteccionRem < end_time)
            ]
        else:
            print("Los tiempos de detección para REM están fuera de los límites de la lista")
            eventos_en_rango_REM = [] 
        """
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
            if  len(eventos_en_rango_spindle) == 0 or len(eventos_en_rango_sw) == 0 or len(eventos_en_rango_EpocaAnterior_spindle) == 0 or len(eventos_en_rango_EpocaAnterior_sw) == 0:
                if  result['GSSC'][1][epoca-1] == 0:
                    print('ES N1')
                    Nueva_anotacion.append(1)
                elif Porcentaje_de_densidad_alpha_cr > Q75_cr_MaxN2_alpha or  Porcentaje_de_densidad_alpha_fr > Q75_fr_MaxN2_alpha or Porcentaje_de_densidad_alpha_pr > Q75_pr_MaxN2_alpha or Porcentaje_de_densidad_alpha_oc > Q75_oc_MaxN2_alpha:
                    print('ES N1')
                    Nueva_anotacion.append(1)
                #elif Porcentaje_de_densidad_Delta_oc < Q25_oc_MaxN2_Delta or Porcentaje_de_densidad_Delta_fr < Q25_fr_MaxN2_Delta or Porcentaje_de_densidad_Delta_cr < Q25_cr_MaxN2_Delta or Porcentaje_de_densidad_Delta_pr < Q25_pr_MaxN2_Delta:
                #    print('ES N1')
                #    Nueva_anotacion.append(1)
                else:
                    print("No se cumplieron las condiciones específicas para N1, queda como N2")
                    Nueva_anotacion.append(2)
            else :
                Nueva_anotacion.append(2)

        elif etapa == 0:
            # Aca me fijo si la etapa evaluada como W realmente corresponde a W
            if epoca + 1 <  result['GSSC'][1].shape[0]:
                if  result['GSSC'][1][epoca + 1] == 2:
                    Nueva_anotacion.append(1)
                    print('ES N1')
                elif Porcentaje_de_densidad_Theta_cr > Q75_cr_MaxW or Porcentaje_de_densidad_Theta_fr > Q75_fr_MaxW or Porcentaje_de_densidad_Theta_pr > Q75_pr_MaxW or Porcentaje_de_densidad_Theta_oc > Q75_oc_MaxW:
                    Nueva_anotacion.append(1)
                    print('ES N1')
                else:
                    print("No se cumplieron las condiciones específicas para W, queda como W")
                    Nueva_anotacion.append(0)
            else:
                print("No se cumplieron las condiciones específicas para W, queda como W")
                Nueva_anotacion.append(0)

        else:
            # Este else final se ejecuta si ninguna de las condiciones anteriores es verdadera
            Nueva_anotacion.append(result['GSSC'][1][epoca])
                    



    return Nueva_anotacion