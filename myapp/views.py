# Author: Chao-Hsiung Hsu
# Version: 1.0
# Copyright: 2024/4/10, StainAI project, Molecular Imaging Laboratory, Howard University
# Contact: chaohsiung.hsu@howard.edu


from django.shortcuts import render
import os
import pandas as pd
from django.http import JsonResponse
from PIL import Image, ImageDraw
import numpy as np
import base64
from io import BytesIO
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import math

def display_image(request):
    current_dir = os.path.dirname(__file__)
    static_dir = os.path.join(current_dir,  'static', 'myapp')
    
    #static_dir = os.path.join('myapp', 'static', 'myapp')  # <-- Update this line
    image_files = [f for f in os.listdir(static_dir) if f.endswith('.jpg')]
    excel_files = [f for f in os.listdir(static_dir) if f.endswith('.xls')]
    df1 = pd.read_excel(os.path.join(static_dir, excel_files[1]))
    df0 = pd.read_excel(os.path.join(static_dir, excel_files[0]))
 
    fm_threshold = int(request.GET.get('fm_threshold', 0))  # Use 0 as the default value
    min_fm = 200 #df['FM'].min()
    max_fm = 1200 #df['FM'].max()
    
   
    # Filter the data based on the fm_threshold
    df_above_threshold = df1[df1['FM'] >= fm_threshold]
    df_below_threshold = df1[df1['FM'] < fm_threshold]

    # Count the occurrences of each value in the 'C50' column for rows above the threshold
    counts_above_threshold = Counter(df_above_threshold['C50'])

    # Only keep the counts for the specified values
    values = ['R', 'H', 'B', 'A', 'RD', 'HR']
    counts_above_threshold = {key: counts_above_threshold[key] for key in values}

    # Count the total number of rows below the threshold
    total_below_threshold = len(df_below_threshold)

    # Add the total count below the threshold as the 7th bar
    counts_above_threshold['< FM'] = total_below_threshold
    # Create a list of counts and a list of colors
    counts = list(counts_above_threshold.values())
        
                      
    # Initialize slidelimit as a list of lists
    sliderlimit1 = [[0, 0], [0, 0], [0, 0], [0, 0]]  #1 for ctrl, 0 for ca
    sliderlimit1[0][0] = math.floor(df1['CArea'].min())
    sliderlimit1[0][1] = math.ceil(df1['CArea'].max())     
    sliderlimit1[1][0] = math.floor(df1['CPM'].min())
    sliderlimit1[1][1] = math.ceil(df1['CPM'].max())              
    sliderlimit1[2][0] = math.floor(df1['CHSR'].min())
    sliderlimit1[2][1] = math.ceil(df1['CHSR'].max())            
    sliderlimit1[3][0] = math.floor(df1['mDS'].min())
    sliderlimit1[3][1] = math.ceil(df1['mDS'].max())   
           
    sliderlimit0 = [[0, 0], [0, 0], [0, 0], [0, 0]]  # 0 for ca
    sliderlimit0[0][0] = math.floor(df0['CArea'].min())
    sliderlimit0[0][1] = math.ceil(df0['CArea'].max())     
    sliderlimit0[1][0] = math.floor(df0['CPM'].min())
    sliderlimit0[1][1] = math.ceil(df0['CPM'].max()) 
    sliderlimit0[2][0] = math.floor(df0['CHSR'].min())
    sliderlimit0[2][1] = math.ceil(df0['CHSR'].max())            
    sliderlimit0[3][0] = math.floor(df0['mDS'].min())
    sliderlimit0[3][1] = math.ceil(df0['mDS'].max())   
               
    context = {
        'image_files': image_files,
        'excel_files': excel_files,
        'min_fm': int(min_fm),
        'max_fm': int(max_fm),
        'counts': counts,
        'sliderlimit1': sliderlimit1,
        'sliderlimit0': sliderlimit0,
    }
    return render(request, 'display_image.html', context)

def show_mmap(request):
    filename = request.GET.get('filename')
    imageType = request.GET.get('type')
    iparameter = request.GET.get('iparameter')
    fm_threshold = int(request.GET.get('fm_threshold', 0))  # Use 0 as the default value
    
    checkBox_R = request.GET.get('checkBox_R', 'false') == 'true'
    checkBox_H = request.GET.get('checkBox_H', 'false') == 'true'
    checkBox_B = request.GET.get('checkBox_B', 'false') == 'true'
    checkBox_A = request.GET.get('checkBox_A', 'false') == 'true'
    checkBox_RD = request.GET.get('checkBox_RD', 'false') == 'true'
    checkBox_HR = request.GET.get('checkBox_HR', 'false') == 'true'
    checkBox_All = request.GET.get('checkBox_All', 'false') == 'true'
    
    
      
    if not filename:
        return JsonResponse({'error': 'No filename provided'}, status=400)

    current_dir = os.path.dirname(__file__)
    static_dir = os.path.join(current_dir,  'static', 'myapp')
    image_path = os.path.join(static_dir, filename)
    
    
    try:
        # Open the image file
        # img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = Image.open(image_path)
    except IOError:
        return JsonResponse({'error': 'Failed to open image file'}, status=500)
   
    excel_filename = os.path.splitext(filename)[0] + '.xls'
    excel_path = os.path.join(static_dir, excel_filename)
    df = pd.read_excel(excel_path)
    # print(df['CArea'].max())
    
    carea_minth = float(request.GET.get('carea_minth',df['CArea'].min()))
    carea_maxth = float(request.GET.get('carea_maxth',df['CArea'].max()))
    # print(carea_minth, carea_maxth)
    cpm_minth = float(request.GET.get('cpm_minth',df['CPM'].min()))
    cpm_maxth = float(request.GET.get('cpm_maxth',df['CPM'].max()))
    
    chsr_minth = float(request.GET.get('chsr_minth',df['CHSR'].min()))
    chsr_maxth = float(request.GET.get('chsr_maxth',df['CHSR'].max()))
    
    mds_minth = float(request.GET.get('mds_minth',df['mDS'].min()))
    mds_maxth = float(request.GET.get('mds_maxth',df['mDS'].max()))
    # print(cpm_minth, cpm_maxth)
    # print(iparameter)
    # print(chsr_maxth)
    
    idth0 = (df['CArea'].between(carea_minth, carea_maxth)) & (df['CPM'].between(cpm_minth, cpm_maxth)) & (df['CHSR'].between(chsr_minth, chsr_maxth)) & (df['mDS'].between(mds_minth, mds_maxth))
    idth1 = df[idth0].index.tolist()
    bbox_values = df['bbox'].values
    # img.show()
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))

    draw = ImageDraw.Draw(overlay)

    # indices = df.groupby('C50').indices
    if iparameter == 'CArea':
        minth = carea_minth
        maxth = carea_maxth
        # dvalues = df['CArea']
        idth2 = ((df['CHSR'] < chsr_minth) | (df['CHSR'] > chsr_maxth) | (df['CPM'] < cpm_minth) | (df['CPM'] > cpm_maxth) | (df['mDS'] < mds_minth) | (df['mDS'] > mds_maxth))
        dvalues = df.loc[~idth2, 'CArea']
    elif iparameter == 'CPM':
        minth = cpm_minth
        maxth = cpm_maxth
        # dvalues = df['CPM']
        idth2 = ((df['CHSR'] < chsr_minth) | (df['CHSR'] > chsr_maxth) | (df['CArea'] < carea_minth) | (df['CArea'] > carea_maxth) | (df['mDS'] < mds_minth) | (df['mDS'] > mds_maxth))
        dvalues = df.loc[~idth2, 'CPM']
    elif iparameter == 'CHSR':
        minth = chsr_minth
        maxth = chsr_maxth
        # dvalues = df['CHSR']
        idth2 = ((df['CPM'] < cpm_minth) | (df['CPM'] > cpm_maxth) | (df['CArea'] < carea_minth) | (df['CArea'] > carea_maxth) | (df['mDS'] < mds_minth) | (df['mDS'] > mds_maxth))
        dvalues = df.loc[~idth2, 'CHSR']

    elif iparameter == 'mDS':
        minth = mds_minth
        maxth = mds_maxth
        # dvalues = df['mDS']
        idth2 = ((df['CPM'] < cpm_minth) | (df['CPM'] > cpm_maxth) | (df['CArea'] < carea_minth) | (df['CArea'] > carea_maxth) & (df['CHSR'] < chsr_minth) | (df['CHSR'] > chsr_maxth))
        dvalues = df.loc[~idth2, 'mDS']

    idth3 = ~idth2
    idth3=df[idth3].index.tolist()   
    # Define the color for FM values below the threshold
    below_threshold_color = (80, 80, 80, 129)  # 40% transparent gray
    # indicesFMGt = df.loc[df['FM'] > 500].index
    # indicesFMLe = df.loc[df['FM'] <= 500].index
    histdata = {
        'bins': [],
        'counts': [],
        'widths': [],
        'colors': [],
    }
    

    if imageType == 'morphological':
        colors = {
            'R': (102, 204, 0, 128),
            'H': (204, 204, 0, 128),
            'B': (220, 112, 0, 128),
            'A': (204, 0, 0, 128),
            'RD': (0, 210, 210, 128),
            'HR': (0, 0, 204, 128)
        }
#        for i, row in df.iterrows():
        for i, row in df.loc[idth1].iterrows():
            bbox = bbox_values[i]
            bbox = bbox.strip('[]')
            bbox2 = [float(num_str) for num_str in bbox.split()]
            x, y, dx, dy = bbox2
            left = x
            upper = y
            right = x + dx
            lower = y + dy

            # Check if the FM value is below the threshold
            if row['FM'] < fm_threshold:
                color = below_threshold_color
                draw.rectangle([left, upper, right, lower], outline=color, fill=color)
            else:
                color = colors.get(row['C50'], (0, 255, 0, 128))  # Use green as the default color
                if checkBox_All or (checkBox_R and row['C50'] == 'R') or (checkBox_B and row['C50'] == 'B') or (checkBox_H and row['C50'] == 'H') or (checkBox_A and row['C50'] == 'A') or (checkBox_RD and row['C50'] == 'RD') or (checkBox_HR and row['C50'] == 'HR'):
                    draw.rectangle([left, upper, right, lower], outline=color, fill=color)

            
   
        img_with_boxes = Image.alpha_composite(img.convert('RGBA'), overlay)
        
    elif imageType == 'parameterMap':
        # Create a colormap
        cmap = plt.cm.jet
   
        # Normalize the CArea values to the range [0, 1]
        df['value_normalized'] = (dvalues- minth) / (maxth - minth)
        df['value_clipped'] = dvalues.clip(minth, maxth)
        df['value_normalized'] = (df['value_clipped'] - minth) / (maxth - minth)

        for i, row in df.loc[idth3].iterrows():  #in df.iterrows():
            bbox = bbox_values[i]
            bbox = bbox.strip('[]')
            bbox2 = [float(num_str) for num_str in bbox.split()]
            x, y, dx, dy = bbox2
            left = x
            upper = y
            right = x + dx
            lower = y + dy
    
            # Check if the FM value is below the threshold
            if row['FM'] < fm_threshold:
                color = below_threshold_color
                draw.rectangle([left, upper, right, lower], outline=color, fill=color)
            else:
                # Map the normalized CArea value to a color in the colormap
                color = cmap(row['value_normalized'])
                # Convert the color from RGBA float (range [0, 1]) to RGBA integer (range [0, 255])
                color = tuple(int(c * 255) for c in color[:3]) + (128,)  # Add 50% transparency                #color = colors.get(row['C50'], (0, 255, 0, 128))  # Use green as the default color

                if checkBox_All or (checkBox_R and row['C50'] == 'R') or (checkBox_B and row['C50'] == 'B') or (checkBox_H and row['C50'] == 'H') or (checkBox_A and row['C50'] == 'A') or (checkBox_RD and row['C50'] == 'RD') or (checkBox_HR and row['C50'] == 'HR'):
                    draw.rectangle([left, upper, right, lower], outline=color, fill=color)
            
   
        img_with_boxes = Image.alpha_composite(img.convert('RGBA'), overlay)
    
    elif imageType == 'original':
            img_with_boxes = img  # Use the original image without overlay
        
       
            
   # Get the checkboxes that are checked
    checked_boxes = [box for box in ['R', 'H', 'B', 'A', 'RD', 'HR'] if request.GET.get(f'checkBox_{box}', 'false') == 'true']

    # Filter the DataFrame based on the checkboxes
    # if checked_boxes:
        # dvalues = dvalues[df['C50'].isin(checked_boxes)]         
    
    # Filter the DataFrame based on the checkboxes and FM threshold
    if checked_boxes:
        dvalues = dvalues[(df['C50'].isin(checked_boxes)) & (df['FM'] > fm_threshold)]
    else:
        dvalues = dvalues[df['FM'] > fm_threshold]
    
    # Get the jet colormap
    jet = get_cmap('jet',200)
    
  
    # Define a custom function to map values to colors
    def map_values_to_colors(dvalues):
        if dvalues < minth:
            return (0, 0, 0.5, 1)   # jet(0)   
        elif dvalues > maxth:
            return (0.5, 0, 0, 1)  # Return red color
        else:
            # Normalize the value to the range [0, 1]
            normalized_value = (dvalues - minth) / (maxth - minth)
            return jet(normalized_value)
    
    # Map the values to colors
    colors = [map_values_to_colors(dvalues) for dvalues in dvalues]
        
    # print(df['CArea'].max())
    # Calculate the histogram
    hist_counts, bins = np.histogram(dvalues, bins=jet.N)
    # Map the bin edges to colors in the colormap
    bin_colors = [map_values_to_colors(bin) for bin in bins]
    # Convert bin_colors to the range [0, 255] and keep only RGB components
    binRGB_colors = [(int(c[0]*255), int(c[1]*255), int(c[2]*255), c[3]) for c in bin_colors]
      
    # Calculate the width of each bin
    widths = np.diff(bins)
    # plt.clf() 
    # Create a bar plot
    # plt.bar(bins[:-1], hist_counts, width=widths, color=bin_colors[:-1])

    # Show the plot
    # plt.show()

    histdata = {
        'bins': bins[:-1].tolist(),
        'counts': hist_counts.tolist(),
        'widths': widths.tolist(),
        'colors': binRGB_colors,
    }


    # Filter the data based on the fm_threshold
    # df_above_threshold = df[df['FM'] >= fm_threshold]
    # df_below_threshold = df[df['FM'] < fm_threshold]
    # Filter the data based on the fm_threshold and the range of idth1
    # df_above_threshold = df.loc[idth1][df['FM'] >= fm_threshold]
    # df_below_threshold = df.loc[idth1][df['FM'] < fm_threshold]

    df_above_threshold = df.loc[idth1][df.loc[idth1, 'FM'] >= fm_threshold]
    df_below_threshold = df.loc[idth1][df.loc[idth1, 'FM'] < fm_threshold]
    
       
    # Count the occurrences of each value in the 'C50' column for rows above the threshold
    counts_above_threshold = Counter(df_above_threshold['C50'])

    # Only keep the counts for the specified values
    values = ['R', 'H', 'B', 'A', 'RD', 'HR']
    counts_above_threshold = {key: counts_above_threshold[key] for key in values}

    # Count the total number of rows below the threshold
    total_below_threshold = len(df_below_threshold)

    # Add the total count below the threshold as the 7th bar
    counts_above_threshold['< FM'] = total_below_threshold
    # Create a list of counts and a list of colors
    counts = list(counts_above_threshold.values())


    # Convert the image to a base64 string
    buffered = BytesIO()
    img_with_boxes_rgb = img_with_boxes.convert('RGB')
    img_with_boxes_rgb.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

   
   # Initialize slidelimit as a list of lists
   # sliderlimit = [[0, 0], [0, 0], [0, 0], [0, 0]]
   # sliderlimit[0][0] = math.floor(df['CArea'].min())
   # sliderlimit[0][1] = math.ceil(df['CArea'].max())     
   # sliderlimit[1][0] = math.floor(df['CPM'].min())
   # sliderlimit[1][1] = math.ceil(df['CPM'].max())   
                 

    return JsonResponse({'counts': counts,
                         'image': img_str, 
                         'histdata': histdata, 
                         }, safe=False)
    # return JsonResponse({'counts': counts,'image': img_str}, safe=False)
    # return JsonResponse({'image': img_str, 'min_fm': min_fm, 'max_fm': max_fm}, safe=False)