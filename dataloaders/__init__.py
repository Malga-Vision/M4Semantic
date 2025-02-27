from .midair import DataLoaderMidAir as MidAir
from .midair1 import DataLoaderMidAir as MidAir1
#from .midair_origin import DataLoaderMidAir as MidAir
#from .kitti import DataLoaderKittiRaw as KittiRaw
from .tartanair import DataLoaderTartanAir as TartanAir
from .wilduav import DataLoaderWUAV as WildUAV
from .uzh import DataLoaderUZH as UZH
from .topair import DataLoaderTopAir as TopAir
#from .airsim import DataLoaderAirsim as Airsim
from .cityscapes import DataLoaderCityScapes as CityScapes
from .aeroscapes import DataLoaderAeroscapes as Aeroscapes
from .ninja2 import DataLoaderNinja2 as Ninja2
from .udd import DataLoaderUDD as UDD
from .generic import DataloaderParameters

def get_loader(name : str):
    available = {
        "midair"        : MidAir(),
        "midair1"        : MidAir1(),
        #"kitti-raw"     : KittiRaw(),
        "tartanair"   : TartanAir(),
        "wilduav"   : WildUAV(),
        "uzh"   : UZH(),
        "topair" : TopAir(),
        #"airsim" : Airsim(),
        "cityscapes": CityScapes(),
        "aeroscapes": Aeroscapes(),
        "ninja2": Ninja2(),
        "udd": UDD()
    }
    try:
        return available[name]
    except:
        print("Dataloaders available:")
        print(available.keys())
        raise NotImplementedError
