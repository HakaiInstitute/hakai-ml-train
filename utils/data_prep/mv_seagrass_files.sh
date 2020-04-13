# Mount the samba data server
sudo mkdir /mnt/H
sudo mount -t cifs -o user=taylor.denouden,domain=victoria.hakai.org //10.10.1.50/Geospatial /mnt/H

mkdir beaumont_2017 \
bennet_bay_2018 \
cabbage_2017 \
choked_pass_2016 \
choked_pass_2017 \
goose_se_2015 \
goose_sw_2015 \
james_bay_2018 \
koeye_2015 \
koeye_2017 \
koeye_2018 \
koeye_2019 \
lyall_harbour_2018 \
marna_2018 \
mcmullin_north_2015 \
mcmullin_south_2015 \
nalau_2019 \
narvaez_bay_2018 \
pruth_bay_2016 \
pruth_bay_2017 \
selby_cove_2017 \
stirling_bay_2019 \
triquet_bay_2016 \
triquet_north_2016 \
tumbo_2018 \
underhill_2019 \
west_bay_mcnaughton_2019

echo "beaumont_2017 /mnt/H/Internal/RS/UAV/Files/GulfIslands/2017/20170721_BeaumontRedo_U231/Products/GulfIslands_Beaumont_mos_U0231
bennet_bay_2018 /mnt/H/Internal/RS/UAV/Files/GulfIslands/2018/GINPR/JBay_20180614_Prossed_EMW_Orthomosaic/JBay_20180614_Prossed_EMW_Orthomosaic
cabbage_2017 /mnt/H/Internal/RS/UAV/Files/GulfIslands/2017/20170721_Cabbage_U0232/Products/GulfIslands_Cabbage_mos_U0232
choked_pass_2016 /mnt/H/Internal/RS/UAV/Files/Calvert/2016/20160803_choked_U0069/Products/20160803_Calvert_ChokedNorthBeach_georef_MOS_U0069
choked_pass_2017 /mnt/H/Internal/RS/UAV/Files/Calvert/2017/20170530_ChokedPassage_U0172/Products/20170529_Calvert_ChokedNorth_georef_mos_U0172
goose_se_2015 /mnt/H/Internal/RS/UAV/Files/CentralCoast/2015/20150903_GooseSouth_U0020/Products/centralcoast_goosegeoref_mos_U0020
goose_sw_2015 /mnt/H/Internal/RS/UAV/Files/CentralCoast/2015/20150903_GooseSouth_U0020/Products/centralcoast_goosegeoref_mos_U0020
james_bay_2018 /mnt/H/Internal/RS/UAV/Files/GulfIslands/2018/GINPR/JBay_20180614_Prossed_EMW_Orthomosaic/JBay_20180614_Prossed_EMW_Orthomosaic
koeye_2015 /mnt/H/Internal/RS/UAV/Files/CentralCoast/2015/20150905_Koeye_U0022/Products/centralcoast_koeye_mos_U0022
koeye_2017 /mnt/H/Internal/RS/UAV/Files/CentralCoast/2017/20170814_Koeye_Eelgrass_U0267/Products/20170814_CentralCoast_Koeye_Georef_mos_U0267
koeye_2018 /mnt/H/Internal/RS/UAV/Files/CentralCoast/2018/20180617_CentralCoastKoeyeSeagrass_U0438/Products/20180617_CentralCoast_KoeyeSeagrass_GeoRef_MOS_U0438
koeye_2019 /mnt/H/Internal/RS/UAV/Files/CentralCoast/2019/20190804_CentralCoast_KoeyeSeagrass_U0715/Products/20190804_CentralCoast_KoeyeSeagrass_georef_MOS_U0715
lyall_harbour_2018 /mnt/H/Internal/RS/UAV/Files/GulfIslands/2018/GINPR/LyallHarbour/LyallHarbour_Orthomosaic_20180616_EMW
marna_2018 /mnt/H/Internal/RS/UAV/Files/Quadra/2018/20181004_Quadra_MarnaSeagrass_U0562/Products/20181004_Quadra_MarnaSeagrass_georef_MOS_U0562
mcmullin_north_2015 /mnt/H/Internal/RS/UAV/Files/CentralCoast/2015/20150904_McMullins_U0021/Products/20150904_CentralCoast_Mcmullins_Georef_mos_U0021
mcmullin_south_2015 /mnt/H/Internal/RS/UAV/Files/CentralCoast/2015/20150904_McMullins_U0021/Products/20150904_CentralCoast_Mcmullins_Georef_mos_U0021
nalau_2019 /mnt/H/Internal/RS/UAV/Files/CentralCoast/2019/20190731_CentralCoast_NalauSeagrass_U0701/Products/CentralCoast_NalauSeagrass_mos_U0701
narvaez_bay_2018 /mnt/H/Internal/RS/UAV/Files/GulfIslands/2018/GINPR/Narvaez_EG_180618zip/narvaez_180618_v1
pruth_bay_2016 /mnt/H/Internal/RS/UAV/Files/Calvert/2016/20160805_626flats_U0071/Products/calvert_pruthsouth_230auto_mos_NAD83_U0071
pruth_bay_2017 /mnt/H/Internal/RS/UAV/Files/Calvert/2017/20170627_626SoftSediement_U0203/Products/20170626_Calvert_PruthSeagrass_GeoRef_MOS_U0203
selby_cove_2017 /mnt/H/Internal/RS/UAV/Files/CentralCoast/2017/20170814_Koeye_Eelgrass_U0267/Products/20170814_CentralCoast_Koeye_Georef_mos_U0267
stirling_bay_2019 /mnt/H/Internal/RS/UAV/Files/CentralCoast/2019/20190719_CentralCoast_StirlingBay_U0694/Products/20190719_CentralCoast_StirlingBayEelgrass_georef_mos_U0694
triquet_bay_2016 /mnt/H/Internal/RS/UAV/Files/CentralCoast/2016/20160705_triquet_U0060/Products/20160507_CentralCoast_NorthTriquet_GeoRef_MOS_U0060
triquet_north_2016 /mnt/H/Internal/RS/UAV/Files/CentralCoast/2016/20160705_triquet_U0060/Products/20160507_CentralCoast_NorthTriquet_GeoRef_MOS_U0060
tumbo_2018 /mnt/H/Internal/RS/UAV/Files/GulfIslands/2018/GINPR/tumbo_180618/Tumbo_eelgrass_20180618
underhill_2019 /mnt/H/Internal/RS/UAV/Files/CentralCoast/2019/20190731_CentralCoast_UnderhillSeagrass_U0700/Products/20190731_CentralCoast_UnderhillSeagrass_60m_georef_MOS_U0700
west_bay_mcnaughton_2019 /mnt/H/Internal/RS/UAV/Files/CentralCoast/2019/20190801_CentralCoast_McNaughtonWestEelgrass_U0703/Products/CentralCoast_WestBayMcNaughton_mos_U0703" | xargs -n2 -P8 sh -c \
'fname="image"; \
cp -u -v "$2.tif" "./$1/$fname.tif"; \
cp -u -v "$2.tfw" "./$1/$fname.tfw"' sh

echo "beaumont_2017 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/GulfIslands/2017/20170721_GulfIslands_Beamont_eelgrass
bennet_bay_2018 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/GulfIslands/2018/20180615_GulfIslands_BennettBay_eelgrass
cabbage_2017 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/GulfIslands/2017/20170721_GulfIslands_Cabbage_eelgrass
choked_pass_2016 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/Calvert/2016/20160803_Calvert_ChokedPass_eelgrass
choked_pass_2017 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/Calvert/2017/20170530_Calvert_ChokedPass_eelgrass
goose_se_2015 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/CentralCoast/2015/20150903_CentralCoast_GooseSE_eelgrass
goose_sw_2015 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/CentralCoast/2015/20150903_CentralCoast_GooseSW_eelgrass
james_bay_2018 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/GulfIslands/2018/20180614_GulfIslands_JamesBay_eelgrass
koeye_2015 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/CentralCoast/2015/20150905_CentralCoast_Koeye_eelgrass
koeye_2017 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/CentralCoast/2017/20170814_CentralCoast_Koeye_eelgrass
koeye_2018 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/CentralCoast/2018/20180617_CentralCoast_Koeye_eelgrass
koeye_2019 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/CentralCoast/2019/20190804_CentralCoast_Koeye_eelgrass
lyall_harbour_2018 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/GulfIslands/2018/20180616_GulfIslands_LyallHarbour_eelgrass
marna_2018 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/Quadra/2018/20181004_Quadra_Marna_eelgrass
mcmullin_north_2015 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/CentralCoast/2015/20150904_CentralCost_McMullinNorth_eelgrass
mcmullin_south_2015 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/CentralCoast/2015/20150904_CentralCoast_McMullinSouth_eelgrass
nalau_2019 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/CentralCoast/2019/20190731_CentralCoast_Nalau_eelgrass
narvaez_bay_2018 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/GulfIslands/2018/20180618_GulfIslands_NarvaezBay_eelgrass
pruth_bay_2016 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/Calvert/2016/20160805_Calvert_PruthBay_eelgrass
pruth_bay_2017 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/Calvert/2017/20170626_Calvert_PruthBay_eelgrass
selby_cove_2017 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/GulfIslands/2017/20170720_GulfIslands_SelbyCove_eelgrass
stirling_bay_2019 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/CentralCoast/2019/20190719_CentralCoast_StirlingBay_eelgrass
triquet_bay_2016 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/CentralCoast/2016/20160705_CentralCoast_TriquetBay_eelgrass
triquet_north_2016 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/CentralCoast/2016/20160705_CentralCoast_TriquetBay_eelgrass
tumbo_2018 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/GulfIslands/2018/20180618_GulfIslands_Tumbo_eelgrass
underhill_2019 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/CentralCoast/2019/20190731_CentralCoast_Underhill_eelgrass
west_bay_mcnaughton_2019 /mnt/H/Internal/GIS/Vector/Marine/Marine_Biology/Eelgrass/CentralCoast/2019/20190801_CentralCoast_WestBayMcNaughton_eelgrass" | xargs -n2 -P8 sh -c \
'fname="seagrass"; \
cp -u -v "$2.dbf" "./$1/$fname.dbf"; \
cp -u -v "$2.shp" "./$1/$fname.shp"; \
cp -u -v "$2.shx" "./$1/$fname.shx"' sh

sudo umount /mnt/H