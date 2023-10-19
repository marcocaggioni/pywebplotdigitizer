from typing import Dict, List, Tuple

import tarfile
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
import io
import cv2
import base64
import matplotlib.pyplot as plt
import numpy as np


from pydantic import BaseModel, validator
from typing import Optional, List

# Define Pydantic model for WPDV4

class Axis(BaseModel):
    name: str
    page: Optional[int]
    type: str
    isLogX: Optional[bool]
    isLogY: Optional[bool]
    isLog: Optional[bool]
    isRotated: Optional[bool]
    isRotated: Optional[bool]
    isDegrees: Optional[bool]
    isClockwise: Optional[bool]
    isRange100: Optional[bool]
    scaleLength: Optional[float]
    unitString: Optional[str]
    calibrationPoints: List[dict]
    
    @validator('type')
    def validate_type(cls, v):
        if v != 'XYAxes':
            raise ValueError('Only implemented the XYAxes type, cannot parse other types')
        return v

class Dataset(BaseModel):
    name: str
    page: Optional[int]
    axesName: str
    metadataKeys: List[str]
    data: List[dict]
    autoDetectionData: Optional[dict]

class Measurement(BaseModel):
    type: str
    data: List[dict]

class WPDV4(BaseModel):
    version: List[int]
    axesColl: List[Axis]
    datasetColl: List[Dataset]
    measurementColl: List[Measurement]
    b64im: Optional[str]

    @property
    def im(self):
        try:
            return base64_to_im(self.b64im)
        except:
            print('No Figure image available')

    def to_records(self):
        records_list= []
        for dataset in self.datasetColl:
            records=dataset.data
            for record in records:
                record['data_x']=record['value'][0]
                record['data_y']=record['value'][1]
                record['dataset_name']= dataset.name
                record['ax_name']= dataset.axesName
            records_list+=records
        return records_list

    def to_table(self):
        table_list= []
        for dataset in self.datasetColl:
            table = pd.DataFrame.from_records(dataset.data)
            table[['data_x', 'data_y']] = pd.DataFrame(table['value'].tolist(), index=table.index)
            table['dataset_name'] = dataset.name
            table['ax_name'] = dataset.axesName
            table_list.append(table)

        full_table=pd.concat(table_list)
        return full_table[['ax_name','dataset_name', 'data_x', 'data_y','x','y']]
    
    def describe(self):
        table = self.to_table()

        try:
            im = base64_to_im(self.b64im)
        except:
            print('No image available')
            fig2, ax2 = plt.subplots(1, 1);
            
            for axes in self.axesColl:
                for dataset in [dataset for dataset in self.datasetColl if dataset.axesName==axes.name]:
                    condition=(table['ax_name']==axes.name) & (table['dataset_name']==dataset.name)
                        
                    ax2.plot(table['x'][condition], -table['y'][condition], 'o', mfc='none', label=dataset.name);
            x_range = ax2.get_xlim()
            y_range = ax2.get_ylim()

            x_range_length = int(x_range[1] - x_range[0])
            y_range_length = int(y_range[1] - y_range[0])

            # unfortunatly we don't know the pixel size of the image used to digitized the data unless we have the image itself
            # fig2.canvas.draw();
            # im = np.array(fig2.canvas.renderer.buffer_rgba());
            
            plt.close(fig2)

            # so we make a placeholder image with a risonable size based on the pixel position of the axes
            im=np.full((x_range_length, y_range_length, 3), 255)
        
        fig_list=[]
        for axes in self.axesColl:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(axes.name)
            
            for dataset in [dataset for dataset in self.datasetColl if dataset.axesName==axes.name]:
                
                    condition=(table['ax_name']==axes.name) & (table['dataset_name']==dataset.name)
                    
                    ax[0].imshow(im)
                    ax[0].plot(table['x'][condition], table['y'][condition], 'o', mfc='none', label=dataset.name)
                    ax[0].axis('off')


                
    
                    ax[1].plot(table['data_x'][condition], table['data_y'][condition], 'o', label=dataset.name)
                    
                    if axes.isLogX:
                        ax[1].set_yscale('log')
                    if axes.isLogY:
                        ax[1].set_xscale('log')
                        
                    ax[1].set_xlabel('data_x')
                    ax[1].set_ylabel('data_y')
                    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
            fig_list.append(fig)
            
        return fig_list
        
    @classmethod
    def from_tar(cls, tar_file_path):
        with tarfile.open(tar_file_path) as tar:
            im = get_dataimage(tar_file_path, from_tar=True)
            datadict=get_datadict(tar_file_path)

        cls=cls.parse_obj(datadict)
        
        png_img = cv2.imencode('.png', im)
        b64_string = base64.b64encode(png_img[1]).decode('utf-8')
        cls.b64im = b64_string

        return cls

    @classmethod
    def from_json(cls, json_file_path, image_file_path=None):
        with open(json_file_path,'r') as file:
            data = json.load(file)
        cls=cls.parse_obj(data)

        if image_file_path is not None:
            
            with open(image_file_path, 'r') as file:
                im = get_dataimage(tar_file_path, from_tar=False)
                png_img = cv2.imencode('.png', im)
                b64_string = base64.b64encode(png_img[1]).decode('utf-8')
                cls.b64im = b64_string
                
        return cls


    def to_tar(self, tar_file_name):
        subfolder=tar_file_name.split('.')[0]
    
        binary_data=base64.b64decode(self.b64im)
        info=tarfile.TarInfo(f'{subfolder}/figure.png')
        info.size=len(binary_data)
        
        with tarfile.open(tar_file_name, mode='w') as tar:
            tar.addfile(info, fileobj=io.BytesIO(binary_data))
    
        json_string= self.json().encode('utf-8')
    
        info=tarfile.TarInfo(f'{subfolder}/wpd.json')
        info.size=len(json_string)
        
        with tarfile.open(tar_file_name, mode='a') as tar:
            tar.addfile(info, fileobj=io.BytesIO(json_string))
    
        json_string=json_string=json.dumps({"version":[4,0],"json":"wpd.json","images":["figure.png"]}).encode('utf-8')
        info=tarfile.TarInfo(f'{subfolder}/info.json')
        info.size=len(json_string)
    
        with tarfile.open(tar_file_name, mode='a') as tar:
            tar.addfile(info, fileobj=io.BytesIO(json_string))

    def to_json(self, json_file_path):
        
        with open(json_file_path,'w') as file:
            file.write(self.json())


def get_dataimage(file_path, from_tar=False):
    if from_tar:
        with tarfile.open(file_path) as tar:
            tar_dict={}
            for file in tar.getmembers():
                tar_dict[file.name.split('/')[-1]] = tar.extractfile(file)
    
            info = json.load(tar_dict['info.json'])
            im = plt.imread(tar_dict[info['images'][0]])
    else:
        im = plt.imread(file_path)
            
    im=cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    return im

def get_datadict(wpd_file: Path) -> Dict:
    with tarfile.open(wpd_file) as tar:
        tar_dict: Dict[str, tarfile.TarFile] = {}
        for file in tar.getmembers():
            tar_dict[file.name.split('/')[-1]] = tar.extractfile(file)

        info = json.load(tar_dict['info.json'])
        data = json.load(tar_dict[info['json']])
    return data

def im_to_base64(im):
    png_img = cv2.imencode('.png', im)
    b64_string = base64.b64encode(png_img[1]).decode('utf-8')
    return b64_string

def base64_to_im(b64_string):
    img_array=np.asarray(bytearray(io.BytesIO(base64.b64decode(b64_string)).read()), dtype=np.uint8)
    im=cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return im*255