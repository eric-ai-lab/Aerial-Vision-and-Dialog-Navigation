import json
import os
import numpy as np
import random
from collections import defaultdict
import cv2 
import torch
import shapely
import shapely.geometry
from shapely.geometry import Point, Polygon, LineString, MultiPoint
from shapely.ops import nearest_points


def compute_iou(a, b):
    a = np.array(a)  # quadrilateral two-dimensional coordinate representation
    poly1 = Polygon(
        a).convex_hull  # python quadrilateral object, will automatically calculate four points, the last four points in the order of: top left bottom right bottom right top left top
    # print(Polygon(a).convex_hull)  # you can print to see if this is the case

    b = np.array(b)
    poly2 = Polygon(b).convex_hull
    # print(Polygon(b).convex_hull)

    union_poly = np.concatenate((a, b))  # Merge two box coordinates to become 8*2
    # print(union_poly)
    # print(MultiPoint(union_poly).convex_hull)  # contains the smallest polygon point of the two quadrilaterals
    if not poly1.intersects(poly2):  # If the two quadrilaterals do not intersect
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # intersection area
            # print(inter_area)
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            # print(union_area)
            if union_area == 0:
                iou = 0
            # iou = float(inter_area)/(union_area-inter_area)  #wrong
            iou = float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
            # The source code gives two ways to calculate IOU, the first one is: intersection part / area of the smallest polygon containing two quadrilaterals
            # The second one: intersection/merge (common way to calculate IOU of rectangular box)
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

def get_direction(start,end):
    vec=np.array(end) - np.array(start)
    _angle = 0
    #          90
    #      135    45
    #     180  .    0
    #      225   -45 
    #          270
    if vec[1] > 0: # lng is postive
        _angle = np.arctan(vec[0]/vec[1]) / 1.57*90
    elif vec[1] < 0:
        _angle = np.arctan(vec[0]/vec[1]) / 1.57*90 + 180
    else:
        if np.sign(vec[0]) == 1:
            _angle = 90
        else:
            _angle = 270
    _angle = (360 - _angle+90)%360
    return _angle

def name_the_direction(_angle):
        if _angle > 337.5 or _angle<22.5:
            return 'north'
        elif np.abs(_angle - 45)<=22.5:
            return 'northeast'
        elif np.abs(_angle -135)<=22.5:
            return 'southeast'
        elif np.abs(_angle - 90)<=22.5:
            return 'east'
        elif np.abs(_angle - 180)<=22.5:
            return 'south'
        elif np.abs(_angle - 315)<=22.5:
            return 'northwest'
        elif np.abs(_angle - 225)<=22.5:
            return 'southwest'
        elif np.abs(_angle - 270)<=22.5:
            return 'west'
class ANDHNavBatch(torch.utils.data.IterableDataset):
    def __init__(self, anno_dir, dataset_dir, splits, tokenizer=None, max_instr_len=512,
        batch_size=64, seed=0, full_traj = False):
        self.dataset_dir = dataset_dir
        self.data = []
        for split in splits:
            new_data = json.load(open(os.path.join(anno_dir, '%s_dataset.json'%split)))
            # # Debug!!!
            # new_data = new_data[:1]
            if full_traj == False:
                for item in new_data:  
                    

                    item['angle'] = round(item['angle']) % 360
                    
                    for i in range(len(item['gt_path_corners'])):
                        item['gt_path_corners'][i] = np.array(item['gt_path_corners'][i]) 
                    item['instructions'] = item['instructions'].lower()
                    item['pre_dialogs'] = ' '.join(item['pre_dialogs']).lower()
                    self.data.append(item)


            # elif full_traj == True:
            #     map_list = set()
            #     for i in range(len(new_data)):
            #         map_list.add(new_data[i]['map_name'])
                
            #     val_map_list = np.array(list(map_list))

            #     for i in range(len(val_map_list)):
            #         sub_trajs_in_single_map = []
            #         traj_idx_in_one_map = set()
            #         for j in range(len(new_data)):
            #             if new_data[j]['map_name'] == val_map_list[i]:
            #                 sub_trajs_in_single_map.append(new_data[j])
            #                 traj_idx_in_one_map.add(new_data[j]['route_index'].split('_')[0])
                            
            #         for traj_idx in traj_idx_in_one_map:
                        
            #             traj_data_json = {}

            #             # get the starting sub_traj
            #             for j in range(len(sub_trajs_in_single_map)):
            #                 if traj_idx == sub_trajs_in_single_map[j]['route_index'].split('_')[0] and\
            #                     str(1) == sub_trajs_in_single_map[j]['route_index'].split('_')[1]:
            #                     traj_data_json = sub_trajs_in_single_map[j]
            #                     traj_data_json['instructions'] = sub_trajs_in_single_map[j]['instructions']
            #                     traj_data_json['angle'] = round(sub_trajs_in_single_map[j]['angle']) % 360
            #                     break
                                
            #             k = 1
            #             while 1:
            #                 k += 1
            #                 if traj_data_json['last_round_idx']<k:
            #                     break
            #                 for j in range(len(sub_trajs_in_single_map)):
            #                     if traj_idx == sub_trajs_in_single_map[j]['route_index'].split('_')[0] and\
            #                         str(k) == sub_trajs_in_single_map[j]['route_index'].split('_')[1]:
                                    
            #                         assert (traj_data_json['lng_ratio'] == sub_trajs_in_single_map[j]['lng_ratio'])
            #                         assert len(traj_data_json['attention_list']) <= len(sub_trajs_in_single_map[j]['attention_list'])
                                    
            #                         traj_data_json['instructions'] += ' [SEP] ' \
            #                                                          +'facing ' + name_the_direction(sub_trajs_in_single_map[j]['angle']) \
            #                                                          + sub_trajs_in_single_map[j]['instructions']
            #                         traj_data_json['attention_list'] = sub_trajs_in_single_map[j]['attention_list'] # last sub-traj attention includes all att in previous sub-trajs
            #                         traj_data_json['gt_path_corners'] += sub_trajs_in_single_map[j]['gt_path_corners']
                                    
            #                         break
            #             for i in range(len(traj_data_json['gt_path_corners'])):
            #                 traj_data_json['gt_path_corners'][i] = np.array(traj_data_json['gt_path_corners'][i]) + np.array([random.random()*1e-7,random.random()*1e-7]) # add noise less than 10cm
                        
            #             des = np.array(traj_data_json['destination'])
            #             mean_des = np.mean(des,axis=0)
            #             best_width = max(max(np.linalg.norm(des[0] - des[1]), np.linalg.norm(des[2] - des[1])), 40/11.13/1e4)

            #             best_goal_view_area = np.array([[mean_des[0]-best_width/2, mean_des[1]-best_width/2],
            #                                             [mean_des[0]-best_width/2, mean_des[1]+best_width/2],
            #                                             [mean_des[0]+best_width/2, mean_des[1]+best_width/2],
            #                                             [mean_des[0]+best_width/2, mean_des[1]-best_width/2]])
                        
            #             traj_data_json['gt_path_corners'].append(best_goal_view_area)

            #             self.data.append(traj_data_json)

            print('%s loaded with %d instructions, using splits: %s' % (
                self.__class__.__name__, len(new_data), split))

        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self.map_batch = {}
        self.attention_map_batch = {}

    def size(self):
        return len(self.data)

    # TODO: find where it is used and then write it
    # def _get_gt_trajs(self, data):
    #     return {x['instr_id']: (x['scan'], x['end_panos']) for x in data if 'end_panos' in x}

    def gps_to_img_coords(self, gps, ob):

        gps_botm_left = ob['gps_botm_left']
        gps_top_right = ob['gps_top_right']
        lng_ratio = ob['lng_ratio']
        lat_ratio = ob['lat_ratio']
        
        return int(round((gps[1] - gps_botm_left[1]) / lat_ratio)), int(round((gps_top_right[0] - gps[0]) / lat_ratio))


    def next_batch(self):
        
        batch_size = self.batch_size
        
        for ix in range(0, len(self.data), batch_size):
            batch = self.data[ix: ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                ix = batch_size - len(batch)
                batch += self.data[:ix]
            
            self.batch = batch

            # Preload all maps used in the batch and clear maps that will not be used.
            used_map_names = []
            for i in range(batch_size):
                used_map_names.append(self.batch[i]['map_name'])
                if not used_map_names[-1] in self.map_batch.keys():
                    im = cv2.imread(os.path.join(self.dataset_dir, used_map_names[-1] + '.tif'), 1)
                    # print('Read map: ', os.path.join(self.dataset_dir, used_map_names[-1] + '.tif'))
                    lng_ratio = self.batch[i]['lng_ratio']
                    lat_ratio = self.batch[i]['lat_ratio']
                    im_resized = cv2.resize(im, (int(im.shape[1]*lng_ratio/lat_ratio ),im.shape[0]), interpolation = cv2.INTER_AREA) # ratio_all = lat_ratio
                    self.map_batch[used_map_names[-1]] = im_resized
                    
                    attention_map = np.zeros((im_resized.shape[0], im_resized.shape[1], 3), np.uint8)
                    for j in range(len(self.batch[i]['attention_list'])):
                        cv2.circle(attention_map, center=self.gps_to_img_coords(self.batch[i]['attention_list'][j][0], 
                                                                                self.batch[i]),
                                radius = self.batch[i]['attention_list'][j][1],
                                color= (255,255,255),
                                thickness=-1) # fill the circle
                    self.attention_map_batch[used_map_names[-1]] = attention_map
                    # cv2.imwrite('../reshaped_' + used_map_names[-1] +'.jpg', im_resized)
                    # cv2.imwrite('../att_' + used_map_names[-1] +'.jpg', attention_map)
            to_be_deleted = []
            for k in self.map_batch:
                if not k in used_map_names:
                    to_be_deleted.append(k)
            for k in to_be_deleted:
                del self.map_batch[k]
                del self.attention_map_batch[k]

            # Get the max instruction length
            max_instruction_length = 0
            for i in range(batch_size):
                if len(self.batch[i]['instructions']) > max_instruction_length:
                    max_instruction_length = len(self.batch[i]['instructions'])
            self.max_instruction_length = max_instruction_length
            
            yield used_map_names
    def __iter__(self):
        return self.next_batch()
    
    # TODO: provide whole environment per time step
    def _get_obs(self, corners=None, directions=None, t=None, shortest_teacher=False):
        obs = []

        for i in range(self.batch_size):

            item = self.batch[i]
            if t == None:
                t_input = 0
            else:
                if t<len(item['gt_path_corners']):
                    t_input = t
                else:
                    t_input = len(item['gt_path_corners']) - 1
            if corners is None:
                view_area_corners = item['gt_path_corners'][t_input]
            else:
                view_area_corners = corners[i]

            # generate view area
            width = 224
            height = 224
            dst_pts = np.array([[0, 0],
                                [width - 1, 0],
                                [width - 1, height - 1],
                                [0, height - 1]], dtype="float32")

            view_area_corners = np.array(view_area_corners)
            img_coord_view_area_corners = view_area_corners
            for xx in range(view_area_corners.shape[0]):
                img_coord_view_area_corners[xx] = self.gps_to_img_coords(view_area_corners[xx], item)
            img_coord_view_area_corners = np.array(img_coord_view_area_corners, dtype="float32")
            
            # the perspective transformation matrix
            M = cv2.getPerspectiveTransform(img_coord_view_area_corners, dst_pts)

            # directly warp the rotated rectangle to get the straightened rectangle
            im_view = cv2.warpPerspective(self.map_batch[item['map_name']], M, (width, height))

            gt_saliency = cv2.warpPerspective(self.attention_map_batch[item['map_name']], M, (width, height))
            gt_saliency = np.asarray(cv2.cvtColor(gt_saliency, cv2.COLOR_BGR2GRAY))/255
            
            
            obs.append({
                'map_name' : item['map_name'],
                'map_size' : self.map_batch[item['map_name']].shape,
                'route_index': item['route_index'],
                                
                'gps_botm_left' : item['gps_botm_left'],
                'gps_top_right' : item['gps_top_right'],

                'lng_ratio' : item['lng_ratio'],
                'lat_ratio' : item['lat_ratio'],
                'starting_angle': item['angle'],
                'current_view' : im_view, 
                'gt_saliency': gt_saliency, 

                # 'instr_encoding': item['instruction'],
                # 'teacher' : self._teacher_path_action(state, item['path'], t=t, shortest_teacher=shortest_teacher),
                'gt_path_corners' : item['gt_path_corners'],
                'view_area_corners': view_area_corners,

                # 'distance': np.linalg.norm(np.mean(view_area_corners, axis=0) - np.mean(item['gt_path_corners'][-1], axis=0)),

                'instructions': item['instructions'],
                'pre_dialogs': item['pre_dialogs']
            })
            
            # TODO: what to use for a2c reward?
            # A3C reward. There are multiple gt end viewpoints on REVERIE. 
            # if 'end_panos' in item:
            #     min_dist = np.inf
            #     for end_pano in item['end_panos']:
            #         min_dist = min(min_dist, self.shortest_distances[scan][viewpoint][end_pano])
            # else:
            #     min_dist = 0
            # obs[-1]['distance'] = min_dist

        
        return obs

    ############### Nav Evaluation ###############
    def _eval_item(self, gt_path, gt_corners, path, corners, progress):
        
        def get_direction(start,end):
            vec=np.array(end) - np.array(start)
            _angle = 0
            #          90
            #      135    45
            #     180  .    0
            #      225   -45 
            #          270
            if vec[1] > 0: # lng is postive
                _angle = np.arctan(vec[0]/vec[1]) / 1.57*90
            elif vec[1] < 0:
                _angle = np.arctan(vec[0]/vec[1]) / 1.57*90 + 180
            else:
                if np.sign(vec[0]) == 1:
                    _angle = 90
                else:
                    _angle = 270
            _angle = (360 - _angle+90)%360
            return _angle

        def path_fid(exec_path, gt_path):
            div_sum = 0
            for i in range(len(exec_path)):
                poly = LineString(gt_path)
                point = Point(exec_path[i])
                p1, p2 = nearest_points(poly, point)
                div_sum += np.linalg.norm(np.array(p1.coords) - exec_path[i])
            return div_sum/len(exec_path)
        
        scores = {}

        # start_pano = path[0]
        # end_panos = set(end_panos)
        # shortest_distances = self.shortest_distances[scan]

        # scores['trajectory_steps'] = len(path) - 1
        

        scores['trajectory_lengths'] = np.sum([np.linalg.norm(a-b) for a, b in zip(path[:-1], path[1:])])
        scores['trajectory_lengths'] = scores['trajectory_lengths']*11.13*1e4
        gt_whole_lengths =  np.sum([np.linalg.norm(a-b) for a, b in zip(gt_path[:-1], gt_path[1:])])*11.13*1e4
        gt_net_lengths =  np.linalg.norm(gt_path[0] - gt_path[-1]) *11.13*1e4


        scores['iou'] = progress[-1]

        # Recording verification: Passed
        # iou = compute_iou(corners[-1], gt_corners[-1]）
        # if scores['iou'] - iou > 0.05:
        #     print('!')


        scores['gp'] = gt_net_lengths - \
                      np.linalg.norm(path[-1] - gt_path[-2])*11.13*1e4
        scores['3d_gp'] = scores['gp']*scores['iou']
        scores['oracle_gp'] = gt_net_lengths - \
                      np.min([np.linalg.norm(path[x] - gt_path[-1]) for x in range(len(path)) ])*11.13*1e4

        
        scores['ATD'] = (path_fid(path,gt_path)*11.13*1e4 + 0.1) / (scores['oracle_gp'] +0.1)* scores['trajectory_lengths'] *100


        # print(len(corners), len(gt_path))
        # print("End corner: ",corners[-1])
        
        # print(gt_path[-1])
  
        # navigation: success is to arrive to a viewpoint in end_panos
        # print(dist)
        if len(path) > 1 and len(gt_path)>1:
            scores['direction_diff_initial'] = min(
                np.abs(get_direction(path[0], path[1]) - get_direction(gt_path[0], gt_path[1])),
                360 - np.abs(get_direction(path[0], path[1]) - get_direction(gt_path[0], gt_path[1])))
            scores['direction_diff'] = min(
                np.abs(get_direction(path[0], path[-1]) - get_direction(gt_path[0], gt_path[-1])),
                360 - np.abs(get_direction(path[0], path[-1]) - get_direction(gt_path[0], gt_path[-1])))
        elif progress[0] >0.4: # at most should be 0.5
            scores['direction_diff_initial'] = 0.
            scores['direction_diff'] = 0.
        else:
            scores['direction_diff_initial'] = 180.
            scores['direction_diff'] = 180.
        
        scores['less_30_rate_direction_diff_initial'] = float(scores['direction_diff_initial']<30)
        scores['less_30_rate_direction_diff'] = float(scores['direction_diff']<15)
                
        scores['success'] = float(progress[-1] >= 0.4)
        _center = np.mean(gt_corners[-1], axis=0) 
        _point = Point(_center)
        _poly = Polygon(np.array(corners[-1]))
        if not _poly.contains(_point):
            scores['success'] = float(0)
        
        _center = np.mean(corners[-1], axis=0) 
        _point = Point(_center)
        _poly = Polygon(np.array(gt_corners[-1]))
        if not _poly.contains(_point):
            scores['success'] = float(0)


        scores['oracle_success'] = float(any(np.array(progress) > 0.4))
        scores['gt_length'] = gt_whole_lengths
        scores['spl'] = scores['success'] * gt_net_lengths / max(scores['trajectory_lengths'], gt_net_lengths, 0.01)

        # gp_max = (np.linalg.norm( np.mean(obs[i]['gt_path_corners'][0], axis=0) - \
        #                                                 np.mean(obs[i]['gt_path_corners'][-1], axis=0))*11.13*1e4\
        #              + max( 400 - np.linalg.norm(obs[i]['gt_path_corners'][-1][0] - obs[i]['gt_path_corners'][-1][1])*11.13*1e4 ,\
        #                     np.linalg.norm(obs[i]['gt_path_corners'][-1][0] - obs[i]['gt_path_corners'][-1][1])*11.13*1e4 - 40)    ) 





        return scores

    def eval_metrics(self, preds, human_att_eval = False):
        ''' Evaluate each agent trajectory based on how close it got to the goal location 
        the path contains [view_id, angle, vofv]'''
        # print('eval %d predictions' % (len(preds)))

        metrics = defaultdict(list)
        if human_att_eval == True:
            for k in preds.keys():
                if 'human_att_performance' in preds[k].keys():
                    metrics['human_att_performance']+=preds[k]['human_att_performance']
                    nss = np.mean(preds[k]['nss'])
                    if nss == nss:
                        metrics['nss'].append(nss)
            metrics['human_att_performance'] = np.mean(metrics['human_att_performance'], axis=0)
            metrics['nss'] = np.mean(metrics['nss'])
            if metrics['nss'] == metrics['nss']: 
                avg_metrics = {"HA_precision": metrics['human_att_performance'][0],
                                "HA_recall": metrics['human_att_performance'][0],
                                "nss": metrics['nss']}
            else: 
                avg_metrics = {"HA_precision": 0,
                                "HA_recall": 0,
                                "nss":0}
            return avg_metrics, metrics

        for k in preds.keys():
            item = preds[k]
            instr_id = item['instr_id']
            # print(instr_id)
            dia_number = 0
            if 'num_dia' in preds[k].keys():
                dia_number = preds[k]['num_dia']
            traj = [np.mean(x[0], axis = 0) for x in item['path_corners']]      # x = (corners, directions)
            corners = [np.array(x[0]) for x in item['path_corners']]      # x = (corners, directions)
            progress = [x for x in item['gt_progress']]
            gt_corners = [np.array(x) for x in item['gt_path_corners']]
            gt_trajs = [np.mean(x, axis = 0) for x in item['gt_path_corners']]
            
            traj_scores = self._eval_item(gt_trajs, gt_corners, traj, corners, progress)
            for k, v in traj_scores.items():
                if k == 'iou' and traj_scores['success']:
                    metrics[k].append(v)
                else:
                    metrics[k].append(v)

            if dia_number == 1:
                metrics['success_1'].append(traj_scores['success'])  
                metrics['spl_1'].append(traj_scores['spl'])
                metrics['gp_1'].append(traj_scores['gp'])
            elif dia_number == 2:
                metrics['success_2'].append(traj_scores['success'])  
                metrics['spl_2'].append(traj_scores['spl'])
                metrics['gp_2'].append(traj_scores['gp'])
            else:
                metrics['success_else'].append(traj_scores['success'])  
                metrics['spl_else'].append(traj_scores['spl'])
                metrics['gp_else'].append(traj_scores['gp'])
                
            if traj_scores['trajectory_lengths'] > 150:
                metrics['success_long'].append(traj_scores['success'])  
                metrics['spl_long'].append(traj_scores['spl'])
                metrics['gp_long'].append(traj_scores['gp'])
                metrics['ATD_long'].append(traj_scores['ATD'])
            else:
                metrics['success_short'].append(traj_scores['success'])  
                metrics['spl_short'].append(traj_scores['spl'])
                metrics['gp_short'].append(traj_scores['gp'])
                metrics['ATD_short'].append(traj_scores['ATD'])
            metrics['instr_id'].append(instr_id)

        avg_metrics = {
            # 'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'sr': np.mean(metrics['success']) * 100,
            'direct_diff_initial': np.mean(metrics['direction_diff_initial']),
            'direct_diff': np.mean(metrics['direction_diff']),
            'less_30_rate_direction_diff_initial': np.mean(metrics['less_30_rate_direction_diff_initial']),
            'less_30_rate_direction_diff': np.mean(metrics['less_30_rate_direction_diff']),
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'gp': np.mean(metrics['gp']),
            '3d_gp': np.mean(metrics['3d_gp']),
            'oracle_gp': np.mean(metrics['oracle_gp']),
            'gt_length': np.mean(metrics['gt_length']),
            'iou' : np.mean(metrics['iou']),
            'ATD' : np.mean(metrics['ATD']),
            # 'ATD_short' : np.mean(metrics['ATD_short']),
            # 'spl_short': np.mean(metrics['spl_short']) * 100,
            # 'sr_short': np.mean(metrics['success_short']) * 100,
            # 'gp_short': np.mean(metrics['gp_short']),
        }
        if len(metrics['success_1']) != 0:
            avg_metrics['num_1']= len(metrics['success_1'])
            avg_metrics['spl_1']= np.mean(metrics['spl_1']) * 100
            avg_metrics['sr_1']=np.mean(metrics['success_1']) * 100
            avg_metrics['gp_1']=np.mean(metrics['gp_1'])

        if len(metrics['success_2']) != 0:
            avg_metrics['num_2']= len(metrics['success_2'])
            avg_metrics['spl_2']= np.mean(metrics['spl_2']) * 100
            avg_metrics['sr_2']=np.mean(metrics['success_2']) * 100
            avg_metrics['gp_2']=np.mean(metrics['gp_2'])

        if len(metrics['success_else']) != 0:
            avg_metrics['num_else']= len(metrics['success_else'])
            avg_metrics['spl_else']= np.mean(metrics['spl_else']) * 100
            avg_metrics['sr_else']=np.mean(metrics['success_else']) * 100
            avg_metrics['gp_else']=np.mean(metrics['gp_else'])
        
        # if len(metrics['ATD_long']) != 0:
        #     avg_metrics['num_long']= len(metrics['ATD_long'])
        #     avg_metrics['ATD_long']= np.mean(metrics['ATD_long'])
        #     avg_metrics['spl_long']= np.mean(metrics['spl_long']) * 100
        #     avg_metrics['sr_long']=np.mean(metrics['success_long']) * 100
        #     avg_metrics['gp_long']=np.mean(metrics['gp_long'])
        # if len(metrics['ATD_short']) != 0:
        #     avg_metrics['num_short']= len(metrics['ATD_short'])
        #     avg_metrics['ATD_short']= np.mean(metrics['ATD_short'])
        #     avg_metrics['spl_short']= np.mean(metrics['spl_short']) * 100
        #     avg_metrics['sr_short']=np.mean(metrics['success_short']) * 100
        #     avg_metrics['gp_short']=np.mean(metrics['gp_short'])
        return avg_metrics, metrics


