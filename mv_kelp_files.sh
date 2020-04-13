# Mount the samba data server
sudo mkdir /mnt/H
sudo mount -t cifs -o user=taylor.denouden,domain=victoria.hakai.org //10.10.1.50/Geospatial /mnt/H

mkdir nw_calvert_2012 \
nw_calvert_2015 \
choked_pass_2016 \
west_beach_2016 \
mcnaughton_2017

echo "nw_calvert_2012	/mnt/H/Internal/RS/Airborne/Air_Photos/Orthophotos_Processed/Calvert_Island_2012/Calvert_ortho_2012_Web_NAD83 \
nw_calvert_2015	/mnt/H/Internal/RS/UAV/Files/Calvert/2015/20150812_NWCalvertFinal_U0015/Products/calvert_nwcalvert15_CSRS_mos_U0015 \
choked_pass_2016	/mnt/H/Internal/RS/UAV/Files/Calvert/2016/20160803_choked_U0069/Products/20160803_Calvert_ChokedNorthBeach_georef_MOS_U0069 \
west_beach_2016	/mnt/H/Internal/RS/UAV/Files/Calvert/2016/20160804_westbeach_U0070/Products/20160804_Calvert_WestBeach_Georef_mos_U0070 \
mcnaughton_2017	/mnt/H/Internal/RS/UAV/Files/CentralCoast/2017/20170527_McNaughtonGroup_U0168/Products/CentralCoast_McNaughtonGroup_MOS_U0168" | xargs -n2 -P8 sh -c \
'fname="image"; \
cp -u -v "$2.tif" "./$1/$fname.tif"; \
cp -u -v "$2.tfw" "./$1/$fname.tfw"' sh

echo "nw_calvert_2012	/mnt/H/Working/For_Taylor/2012_Kelp/2012_Kelp_RC_1 \
nw_calvert_2015	/mnt/H/Working/For_Taylor/2015_Kelp/2015_U0015_kelp \
choked_pass_2016	/mnt/H/Working/For_Taylor/2016_Kelp/2016_U069_Kelp_RC_1 \
west_beach_2016	/mnt/H/Working/For_Taylor/2016_Kelp/2016_U070_Kelp_RC_1 \
mcnaughton_2017	/mnt/H/Working/For_Taylor/McNaughtons_U0168/McNaughton_kelp" | xargs -n2 -P8 sh -c \
'fname="kelp"; \
cp -u -v "$2.tif" "./$1/$fname.tif"; \
cp -u -v "$2.tfw" "./$1/$fname.tfw"' sh

sudo umount /mnt/H