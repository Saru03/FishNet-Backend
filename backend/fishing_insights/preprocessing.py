from django.conf import settings
import os
from dateutil import parser

class sst_chlorophyll_processing:
    def sst_chlorophyll(date, latitude, longitude):
        dt = parser.parse(date)
        formatted_date = int(dt.strftime("%Y%m%d"))
        sst_file = os.path.join(settings.BASE_DIR, "..","AQUA_MODIS.`${formatted_date}`.L3m.DAY.NSST.sst.4km.NRT.nc`")
        chloro_file = os.path.join(settings.BASE_DIR, "..","AQUA_MODIS.`${formatted_date}`.L3m.DAY.NSST.sst.4km.NRT.nc`")
