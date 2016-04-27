import pandas as pd
import numpy as np
import datetime
import scipy as sc
from __future__ import division
import time
import datetime
from sklearn.cluster import KMeans
import gbq_large
import subprocess


def main():
	start_date = datetime.datetime.now()
	search_date = start_date + datetime.timedelta(-30) 
	week1_query ='''SELECT T1.uid_i as uid_i,ave as ave_f, special_crystal as special_crystal_f, 
	    pve_consumable as pve_consumable_f, upgrade as upgrade_f, premium_hero as premium_hero_f, n_transactions_i,age_i

	FROM


	(SELECT uid_i, s_ave/total as ave, s_special_crystal/total as special_crystal, s_pve_consumable/total as pve_consumable, 
	s_upgrade/total as upgrade, s_premium_hero/total as premium_hero, n_transactions_i
	FROM

	(SELECT uid_i, SUM(ave) as s_ave, SUM(special_crystal) as s_special_crystal, sum(pve_consumable) as s_pve_consumable, sum(upgrade) as s_upgrade,
	sum(premium_hero) as s_premium_hero,
	(SUM(ave) + SUM(special_crystal) +sum(pve_consumable)+sum(upgrade)+sum(premium_hero)) as total, COUNT(*) as n_transactions_i
	FROM

	(SELECT uid_i, data_reason_desc_s,data_reason_pricing_id_s, 
	(case when left(data_reason_pricing_id_s,4) ='ave_' then data_item_q_i else 0 end) as ave,

	(case when data_reason_pricing_id_s LIKE('%crystal%') and data_reason_pricing_id_s not LIKE('%golden%') then data_item_q_i 
	when data_reason_pricing_id_s LIKE('%upsale%') then data_item_q_i 
	when data_reason_pricing_id_s like('rocket%') then data_item_q_i
	else 0 end) as special_crystal,

	(case when data_reason_pricing_id_s LIKE('%golden%') then data_item_q_i 
	when data_reason_pricing_id_s LIKE('%upgrade%') then data_item_q_i 
	when data_reason_pricing_id_s LIKE('%regen%') then data_item_q_i
	when data_reason_pricing_id_s LIKE('%arena%') then data_item_q_i
	when data_reason_pricing_id_s LIKE('%duel%') then data_item_q_i
	when data_reason_pricing_id_s LIKE('%key%') then data_item_q_i 
	when data_reason_pricing_id_s is null then data_item_q_i
	else 0 end) upgrade,


	(case when data_reason_pricing_id_s LIKE('health_potion%') then data_item_q_i
	when data_reason_pricing_id_s LIKE('revive%') then data_item_q_i
	when data_reason_pricing_id_s LIKE('team%') then data_item_q_i
	when data_reason_pricing_id_s LIKE('%questing_pack%') then data_item_q_i
	when data_reason_pricing_id_s LIKE('%booster%') then data_item_q_i
	when data_reason_pricing_id_s LIKE('%pve_refill%') then data_item_q_i
	else 0 end) as pve_consumable,


	(case when data_reason_pricing_id_s LIKE('%premium_hero%') then data_item_q_i else 0 end) premium_hero,


	FROM table_date_range(marvel_production_view.redeemer_transactions,timestamp(\''''+str(search_date)+'''\'),timestamp(\''''+str(start_date)+'''\'))
	where counter_s = 'spend'
	and data_item_n_s ='hc'
	and data_reason_desc_s !='buyGift'
	and data_reason_pricing_id_s !='fte_guaranteed'
	and data_reason_pricing_id_s not LIKE('hero_crystal%')
	and data_reason_pricing_id_s !='alliance_create_cost_b')
	GROUP EACH BY 1)) T1
	JOIN EACH
	(SELECT uid_i, DATEDIFF(timestamp(\''''+str(start_date)+'''\'),time_join_t) as age_i
	FROM marvel_production_view.users

	where time_join_t < timestamp(\''''+str(search_date)+'''\')) T2
	ON T1.uid_i = T2.uid_i

	'''
	df_dimensions_collapsed_w1 = pd.read_gbq(week1_query,project_id='mcoc-bi')
	df_dimensions_collapsed_w1=df_dimensions_collapsed_w1.fillna(0)
	df_dimensions = df_dimensions_collapsed_w1[['ave_f','special_crystal_f','pve_consumable_f','upgrade_f','premium_hero_f']]
	est_c = KMeans(n_clusters=10)
	est_c.cluster_centers_ = np.asarray([[ 0.02694769,  0.06531768,  0.06121219,  0.82539261,  0.02112983],
	       [ 0.05772959,  0.37772436,  0.09730477,  0.40487444,  0.06236684],
	       [ 0.08125626,  0.29389585,  0.42306508,  0.12683245,  0.07495037],
	       [ 0.01135739,  0.08087575,  0.0494629 ,  0.0646581 ,  0.79364585],
	       [ 0.51941725,  0.14303638,  0.15421209,  0.14783146,  0.03550281],
	       [ 0.00832494,  0.91744861,  0.02002415,  0.03100689,  0.02319541],
	       [ 0.06583563,  0.62194053,  0.09732582,  0.12262572,  0.0922723 ],
	       [ 0.08316859,  0.09417081,  0.33420608,  0.44578459,  0.04266993],
	       [ 0.03944744,  0.05819858,  0.79046582,  0.09186975,  0.02001841],
	       [ 0.04018328,  0.35265425,  0.08800917,  0.11709595,  0.40205735]])
	labels_c=est_c.predict(df_dimensions)
	df_dimensions_collapsed_w1['cluster_label_i'] = labels_c
	df_write = df_dimensions_collapsed_w1
	df_write['ave_f'] = df_write.ave_f.apply(lambda x: np.fabs(x))
	df_write['special_crystal_f'] = df_write.special_crystal_f.apply(lambda x: np.fabs(x))
	df_write['pve_consumable_f'] = df_write.pve_consumable_f.apply(lambda x: np.fabs(x))
	df_write['upgrade_f'] = df_write.upgrade_f.apply(lambda x: np.fabs(x))
	df_write['premium_hero_f'] = df_write.premium_hero_f.apply(lambda x: np.fabs(x))
	df_write['_ts_t'] = start_date.strftime('%Y-%m-%d %H:%M:%S')
	filename_str = 'segmentation.csv'
	table_write = 'mcoc-bi:marvel_bi.user_segmentation_historical'+ start_date.strftime('%Y%m%d')
	df_write.to_csv(filename_str,index=False)
	subprocess.call("bq load --source_format=CSV --skip_leading_rows=1 "+table_write+ " " + filename_str + " uid_i:integer,ave_f:float,special_crystal_f:float,pve_consumable_f:float,upgrade_f:float,premium_hero_f:float,n_transactions_i:integer,age_i:integer,cluster_label_i:integer,_ts_t:timestamp",shell=True)

if __name__ == "__main__":
	main()