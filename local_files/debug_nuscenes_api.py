# 主要用于查看nuscnes的api调用方法
# https://www.nuscenes.org/tutorials/nuscenes_tutorial.html
from nuscenes.nuscenes import NuScenes


def debug_get_scene_sample(nusc):
    # 会输出每个scene的基本信息
    # scene-0061, Parked truck, construction, intersectio... [18-07-24 03:28:47]   19s, singapore-onenorth, #anns:4622
    nusc.list_scenes()
    # 场景的数量
    print("scenes num:", len(nusc.scene))

    # 得到一个scene的基本信息,包含first_sample_token、last_sample_token和name等信息
    # {'token': 'cc8c0bf57f984915a77078b10eb33198', 'log_token': '7e25a2c8ea1f41c5b0da1e69ecfa71a2', 'nbr_samples': 39, 'first_sample_token': 'ca9a282c9e77460f8360f564131a8af5', 'last_sample_token': 'ed5fc18c31904f96a8f0dbb99ff069c0', 'name': 'scene-0061', 'description': 'Parked truck, construction, intersection, turn left, following a van'}
    my_scene = nusc.scene[0]
    print(my_scene)

    # 得到这个scene的sample信息
    first_sample_token = my_scene['first_sample_token']
    # {'token': 'ca9a282c9e77460f8360f564131a8af5', 'timestamp': 1532402927647951, 'prev': '', 'next': '39586f9d59004284a7114a68825e8eec', 'scene_token': 'cc8c0bf57f984915a77078b10eb33198', 'data': {'RADAR_FRONT': '37091c75b9704e0daa829ba56dfa0906', 'RADAR_FRONT_LEFT': '11946c1461d14016a322916157da3c7d', 'RADAR_FRONT_RIGHT': '491209956ee3435a9ec173dad3aaf58b', 'RADAR_BACK_LEFT': '312aa38d0e3e4f01b3124c523e6f9776', 'RADAR_BACK_RIGHT': '07b30d5eb6104e79be58eadf94382bc1', 'LIDAR_TOP': '9d9bf11fb0e144c8b446d54a8a00184f', 'CAM_FRONT': 'e3d495d4ac534d54b321f50006683844', 'CAM_FRONT_RIGHT': 'aac7867ebf4f446395d29fbd60b63b3b', 'CAM_BACK_RIGHT': '79dbb4460a6b40f49f9c150cb118247e', 'CAM_BACK': '03bea5763f0f4722933508d5999c5fd8', 'CAM_BACK_LEFT': '43893a033f9c46d4a51b5e08a67a1eb7', 'CAM_FRONT_LEFT': 'fe5422747a7d4268a4b07fc396707b23'}, 'anns': ['ef63a697930c4b20a6b9791f423351da', '6b89da9bf1f84fd6a5fbe1c3b236f809', '924ee6ac1fed440a9d9e3720aac635a0', '91e3608f55174a319246f361690906ba', 'cd051723ed9c40f692b9266359f547af', '36d52dfedd764b27863375543c965376', '70af124fceeb433ea73a79537e4bea9e', '63b89fe17f3e41ecbe28337e0e35db8e', 'e4a3582721c34f528e3367f0bda9485d', 'fcb2332977ed4203aa4b7e04a538e309', 'a0cac1c12246451684116067ae2611f6', '02248ff567e3497c957c369dc9a1bd5c', '9db977e264964c2887db1e37113cddaa', 'ca9c5dd6cf374aa980fdd81022f016fd', '179b8b54ee74425893387ebc09ee133d', '5b990ac640bf498ca7fd55eaf85d3e12', '16140fbf143d4e26a4a7613cbd3aa0e8', '54939f11a73d4398b14aeef500bf0c23', '83d881a6b3d94ef3a3bc3b585cc514f8', '74986f1604f047b6925d409915265bf7', 'e86330c5538c4858b8d3ffe874556cc5', 'a7bd5bb89e27455bbb3dba89a576b6a1', 'fbd9d8c939b24f0eb6496243a41e8c41', '198023a1fb5343a5b6fad033ab8b7057', 'ffeafb90ecd5429cba23d0be9a5b54ee', 'cc636a58e27e446cbdd030c14f3718fd', '076a7e3ec6244d3b84e7df5ebcbac637', '0603fbaef1234c6c86424b163d2e3141', 'd76bd5dcc62f4c57b9cece1c7bcfabc5', '5acb6c71bcd64aa188804411b28c4c8f', '49b74a5f193c4759b203123b58ca176d', '77519174b48f4853a895f58bb8f98661', 'c5e9455e98bb42c0af7d1990db1df0c9', 'fcc5b4b5c4724179ab24962a39ca6d65', '791d1ca7e228433fa50b01778c32449a', '316d20eb238c43ef9ee195642dd6e3fe', 'cda0a9085607438c9b1ea87f4360dd64', 'e865152aaa194f22b97ad0078c012b21', '7962506dbc24423aa540a5e4c7083dad', '29cca6a580924b72a90b9dd6e7710d3e', 'a6f7d4bb60374f868144c5ba4431bf4c', 'f1ae3f713ba946069fa084a6b8626fbf', 'd7af8ede316546f68d4ab4f3dbf03f88', '91cb8f15ed4444e99470d43515e50c1d', 'bc638d33e89848f58c0b3ccf3900c8bb', '26fb370c13f844de9d1830f6176ebab6', '7e66fdf908d84237943c833e6c1b317a', '67c5dbb3ddcc4aff8ec5140930723c37', 'eaf2532c820740ae905bb7ed78fb1037', '3e2d17fa9aa5484d9cabc1dfca532193', 'de6bd5ffbed24aa59c8891f8d9c32c44', '9d51d699f635478fbbcd82a70396dd62', 'b7cbc6d0e80e4dfda7164871ece6cb71', '563a3f547bd64a2f9969278c5ef447fd', 'df8917888b81424f8c0670939e61d885', 'bb3ef5ced8854640910132b11b597348', 'a522ce1d7f6545d7955779f25d01783b', '1fafb2468af5481ca9967407af219c32', '05de82bdb8484623906bb9d97ae87542', 'bfedb0d85e164b7697d1e72dd971fb72', 'ca0f85b4f0d44beb9b7ff87b1ab37ff5', 'bca4bbfdef3d4de980842f28be80b3ca', 'a834fb0389a8453c810c3330e3503e16', '6c804cb7d78943b195045082c5c2d7fa', 'adf1594def9e4722b952fea33b307937', '49f76277d07541c5a584aa14c9d28754', '15a3b4d60b514db5a3468e2aef72a90c', '18cc2837f2b9457c80af0761a0b83ccc', '2bfcc693ae9946daba1d9f2724478fd4']}
    # 'token'这个sample的token信息
    # 'next'这个sample的下个sample的token信息，如果这个sample为该场景最后一个sample，那么next为''
    # 'scene_token'为所属的scene的token信息
    # 'data'对应的传感器数据，包括'CAM_FRONT': 'e3d495d4ac534d54b321f50006683844'等，可以拿到对应传感器对应数据的token
    # 'anns'对应的标注信息: ['ef63a697930c4b20a6b9791f423351da', ....], 可以拿到对应标注信心的token
    my_sample = nusc.get('sample', first_sample_token)

    # ['token', 'timestamp', 'prev', 'next', 'scene_token', 'data', 'anns']
    print(my_sample.keys())

    # 可以看到这个sample的一些基本信息
    # 传感器信息：sample_data_token: e3d495d4ac534d54b321f50006683844, mod: camera, channel: CAM_FRONT
    # ann信息：sample_annotation_token: 2bfcc693ae9946daba1d9f2724478fd4, category: movable_object.barrier
    nusc.list_sample(my_sample['token'])


def debug_show_sample_data(nusc):
    my_scene = nusc.scene[0]
    first_sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', first_sample_token)

    # dict_keys(['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT',
    # 'LIDAR_TOP', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'])
    print(my_sample['data'].keys())

    if 0:
        # 显示某个传感器的数据，同时显示该传感器的标注信息
        for sensor in ['CAM_FRONT', 'LIDAR_TOP']:
            # sensor = 'LIDAR_TOP'
            sensor_data = nusc.get('sample_data', my_sample['data'][sensor])
            nusc.render_sample_data(sensor_data['token'])

    # sample_annotation refers to any bounding box defining the position of an object seen in a sample.
    # All location data is given with respect to the global coordinate system.
    # Let's examine an example from our sample above.
    my_annotation_token = my_sample['anns'][4]
    print("my_annotation_token:", my_annotation_token)
    # my_annotation_token = "0e79e7bed1d543c9a00acfbf90ff60b3"
    my_annotation_metadata = nusc.get('sample_annotation', my_annotation_token)
    # my_annotation_metadata
    print(my_annotation_metadata)
    # 将ann在不同传感器中的标注可视化
    # nusc.render_annotation(my_annotation_token)
    nusc.render_annotation(my_annotation_token, extra_info=True, out_path="./data/" + my_annotation_token)


def debug_show_instance_data(nusc):
    my_scene = nusc.scene[0]
    first_sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', first_sample_token)

    # Object instance are instances that need to be detected or tracked by an AV
    # (e.g a particular vehicle, pedestrian).
    my_instance = nusc.instance[599]
    print("my_instance:", my_instance)
    # {'token': '9cba9cd8af85487fb010652c90d845b5',
    #  'category_token': 'fedb11688db84088883945752e480c2c',
    #  'nbr_annotations': 16,     # 总共16个标注信息
    #  'first_annotation_token': '77afa772cb4a4e5c8a5a53f2019bdba0',
    #  'last_annotation_token': '6fed6d902e5e487abb7444f62e1a2341'}

    # We generally track an instance across different frames in a particular scene.
    # However, we do not track them across different scenes.
    # In this example, we have 16 annotated samples for this instance across a particular scene.

    instance_token = my_instance['token']

    instance_token = "0e79e7bed1d543c9a00acfbf90ff60b3"
    # nusc.render_instance(instance_token)     # 采用这种方式不显示
    # nusc.render_instance(instance_token, extra_info=True, out_path="./22")   # 需要才能显示,默认显示第一个

    # 得到instance的所有ann token
    ann_tokens = nusc.field2token('sample_annotation', 'instance_token', instance_token)

    for i, ann_token in enumerate(ann_tokens):
        # nusc.render_annotation(my_instance['first_annotation_token'])
        nusc.render_annotation(ann_token, extra_info=True, out_path="./data/" + str(i))


def debug_show_category_attribute(nusc):
    # A category is the object assignment of an annotation.
    # Let's look at the category table we have in our database.
    # The table contains the taxonomy of different object categories and also list the subcategories (delineated by a period).
    # human.pedestrian.adult      n= 4765, width= 0.68±0.11, len= 0.73±0.17, height= 1.76±0.12, lw_aspect= 1.08±0.23
    # 包括类别的数量, 长宽高的范围等
    nusc.list_categories()

    #category: {'token': '653f7efbb9514ce7b81d44070d6208c1', 'name': 'movable_object.barrier', 'description': 'Temporary road barrier placed in the scene in order to redirect traffic. Commonly used at construction sites. This includes concrete barrier, metal barrier and water barrier. No fences.', 'index': 9}
    # 包括类别的token和描述等
    print("category:", nusc.category[9])

    # An attribute is a property of an instance that may change throughout different parts of a scene while the category remains the same.
    # Here we list the provided attributes and the number of annotations associated with a particular attribute.
    nusc.list_attributes()

    my_instance = nusc.instance[27]
    first_token = my_instance['first_annotation_token']
    last_token = my_instance['last_annotation_token']
    nbr_samples = my_instance['nbr_annotations']
    print("nbr_samples:", nbr_samples)
    current_token = first_token

    i = 0
    found_change = False
    # attribute为ann标注的一部分，sample_annotation通过next得到该anno的下个标注信息
    while current_token != last_token:
        current_ann = nusc.get('sample_annotation', current_token)
        current_attr = nusc.get('attribute', current_ann['attribute_tokens'][0])['name']

        if i == 0:
            pass
        elif current_attr != last_attr:
            print("Changed from `{}` to `{}` at timestamp {} out of {} "
                  "annotated timestamps".format(last_attr, current_attr, i, nbr_samples))
            found_change = True

        next_token = current_ann['next']
        current_token = next_token
        last_attr = current_attr
        i += 1
        print(i)


def debug_show_visibility_data(nusc):
    # 给出visibility描述,所有可见属性都有那些内容
    print(nusc.visibility)

    anntoken = 'a7d0722bce164f88adf03ada491ea0ba'
    visibility_token = nusc.get('sample_annotation', anntoken)['visibility_token']

    # Visibility: {'description': 'visibility of whole object is between 80 and 100%', 'token': '4', 'level': 'v80-100'}
    # Visibility为ann标注的一部分，
    print("Visibility: {}".format(nusc.get('visibility', visibility_token)))
    nusc.render_annotation(anntoken)

    anntoken = '9f450bf6b7454551bbbc9a4c6e74ef2e'
    visibility_token = nusc.get('sample_annotation', anntoken)['visibility_token']

    print("Visibility: {}".format(nusc.get('visibility', visibility_token)))
    nusc.render_annotation(anntoken)


def debug_sensor_calibrated_sensor_ego_pose(nusc):
    # 得到所有sensor的列表
    # {'token': '725903f5b62f56118f4094b46a4470d8', 'channel': 'CAM_FRONT', 'modality': 'camera'}
    print(nusc.sensor)

    # 某个sample_data的基础信息，包含了sensor的类别
    # {'token': '2ecfec536d984fb491098c9db1404117',
    #  'sample_token': '356d81f38dd9473ba590f39e266f54e5',
    #  'ego_pose_token': '2ecfec536d984fb491098c9db1404117',
    #  'calibrated_sensor_token': 'f4d2a6c281f34a7eb8bb033d82321f79',
    #  'timestamp': 1532402928269133,
    #  'fileformat': 'pcd',
    #  'is_key_frame': False,
    #  'height': 0,
    #  'width': 0,
    #  'filename': 'sweeps/RADAR_FRONT/n015-2018-07-24-11-22-45+0800__RADAR_FRONT__1532402928269133.pcd',
    #  'prev': 'b933bbcb4ee84a7eae16e567301e1df2',
    #  'next': '79ef24d1eba84f5abaeaf76655ef1036',
    #  'sensor_modality': 'radar',
    #  'channel': 'RADAR_FRONT'}
    nusc.sample_data[10]

    # calibrated_sensor
    # 定义是具体传感器在车上的位置
    #calibrated_sensor consists of the definition of a particular sensor (lidar/radar/camera) as calibrated on a particular vehicle.
    print("calibrated_sensor num", len(nusc.calibrated_sensor))
    # {'token': 'f4d2a6c281f34a7eb8bb033d82321f79', 'sensor_token': '47fcd48f71d75e0da5c8c1704a9bfe0a',
    # 'translation': [3.412, 0.0, 0.5], 'rotation': [0.9999984769132877, 0.0, 0.0, 0.0017453283658983088],
    # 'camera_intrinsic': []}
    print(nusc.calibrated_sensor[0])   # 120个参数数据，可能对应不同的采集车，在一个采集车目前有12标定好的传感器

    # ego_pose
    #ego_pose contains information about the location (encoded in translation)
    # and the orientation (encoded in rotation) of the ego vehicle, with respect to the global coordinate system.
    # 将当前车辆的位置转到世界坐标
    print("ego_pose num", len(nusc.ego_pose))   # 31206代表采集了这么多个数据
    # {'token': '5ace90b379af485b9dcb1584b01e7212', 'timestamp': 1532402927814384,
    # 'rotation': [0.5731787718287827, -0.0015811634307974854, 0.013859363182046986, -0.8193116095230444],
    # 'translation': [410.77878632230204, 1179.4673290964536, 0.0]}
    print(nusc.ego_pose[0])


def debug_log_map(nusc):
    # log
    # The log table contains log information from which the data was extracted.
    # A log record corresponds to one journey of our ego vehicle along a predefined route.
    print("Number of `logs` in our loaded database: {}".format(len(nusc.log)))
    # {'token': '7e25a2c8ea1f41c5b0da1e69ecfa71a2', 'logfile': 'n015-2018-07-24-11-22-45+0800',
    # 'vehicle': 'n015', 'date_captured': '2018-07-24', 'location': 'singapore-onenorth',
    # 'map_token': '53992ee3023e5494b90c316c183be829'}
    # 可以得到采集的时间、地点、车辆信息和map信息等
    print(nusc.log[0])

    # map
    # Map information is stored as binary semantic masks from a top-down view.
    print("There are {} maps masks in the loaded dataset".format(len(nusc.map)))
    # 用二值图的形式显示map的形状，一个map中可以包含很多不同的log
    # {'category': 'semantic_prior', 'token': '53992ee3023e5494b90c316c183be829',
    # 'filename': 'maps/53992ee3023e5494b90c316c183be829.png',
    # 'log_tokens': ['0986cb758b1d43fdaa051ab23d45582b', '1c9b302455ff44a9a290c372b31aa3ce', 'e60234ec7c324789ac7c8441a5e49731', '46123a03f41e4657adc82ed9ddbe0ba2', 'a5bb7f9dd1884f1ea0de299caefe7ef4', 'bc41a49366734ebf978d6a71981537dc', 'f8699afb7a2247e38549e4d250b4581b', 'd0450edaed4a46f898403f45fa9e5f0d', 'f38ef5a1e9c941aabb2155768670b92a', '7e25a2c8ea1f41c5b0da1e69ecfa71a2', 'ddc03471df3e4c9bb9663629a4097743', '31e9939f05c1485b88a8f68ad2cf9fa4', '783683d957054175bda1b326453a13f4', '343d984344e440c7952d1e403b572b2a', '92af2609d31445e5a71b2d895376fed6', '47620afea3c443f6a761e885273cb531', 'd31dc715d1c34b99bd5afb0e3aea26ed', '34d0574ea8f340179c82162c6ac069bc', 'd7fd2bb9696d43af901326664e42340b', 'b5622d4dcb0d4549b813b3ffb96fbdc9', 'da04ae0b72024818a6219d8dd138ea4b', '6b6513e6c8384cec88775cae30b78c0e', 'eda311bda86f4e54857b0554639d6426', 'cfe71bf0b5c54aed8f56d4feca9a7f59', 'ee155e99938a4c2698fed50fc5b5d16a', '700b800c787842ba83493d9b2775234a'],
    # 'mask': <nuscenes.utils.map_mask.MapMask object at 0x7f1c48a59090>}
    print(nusc.map[0])


def debug_nuscene_basic(nusc):
    # The NuScenes class holds several tables. Each table is a list of records, and each record is a dictionary.
    # 通过nusc.xxx的方式可以得到某种信息的token key, 然后通过nusc.get(xxx)的方式得到具体的内容
    nusc.category[0]
    cat_token = nusc.category[0]['token']
    nusc.get('category', cat_token)

    nusc.sample_annotation[0]
    nusc.get('visibility', nusc.sample_annotation[0]['visibility_token'])
    one_instance = nusc.get('instance', nusc.sample_annotation[0]['instance_token'])

    # 得到某个instance token对应的所有ann token
    # This returns a list of all sample_annotation records with the 'instance_token' == one_instance['token'].
    # The nusc.field2token() method is generic and can be used in any similar situation. 可以扩张应该用到其他的不同类型token
    ann_tokens = nusc.field2token('sample_annotation', 'instance_token', one_instance['token'])
    ann_tokens_field2token = set(ann_tokens)
    print(len(ann_tokens_field2token))
    print(len(ann_tokens))

    # 方式2, 通过next遍历的方式得到某个instance token对应的所有ann token
    ann_record = nusc.get('sample_annotation', one_instance['first_annotation_token'])
    ann_tokens_traverse = set()
    ann_tokens_traverse.add(ann_record['token'])
    while not ann_record['next'] == "":
        ann_record = nusc.get('sample_annotation', ann_record['next'])
        ann_tokens_traverse.add(ann_record['token'])

    # 间接和直接拿到所需的类别和sensor_record等信息
    catname = nusc.sample_annotation[0]['category_name']
    ann_rec = nusc.sample_annotation[0]
    inst_rec = nusc.get('instance', ann_rec['instance_token'])
    cat_rec = nusc.get('category', inst_rec['category_token'])
    print(catname == cat_rec['name'])

    # Shortcut
    channel = nusc.sample_data[0]['channel']
    # No shortcut
    sd_rec = nusc.sample_data[0]
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])

    print(channel == sensor_record['channel'])

def debug_show_render(nusc):
    # 雷达数据
    # my_sample = nusc.sample[10]
    # nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP', camera_channel='CAM_FRONT')

    # 选择不同的cam
    # nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP', camera_channel='CAM_FRONT_RIGHT')

    # 显示强度
    # nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP', render_intensity=True)

    # 显示毫米波雷达
    # nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='RADAR_FRONT')

    # 其他传感器的显示
    my_sample = nusc.sample[20]
    # CAM_FRONT
    # nusc.render_sample_data(my_sample['data']['CAM_FRONT'])
    # nsweeps 可以设置显示多少个sweeps的数据进行叠加
    # nusc.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=5, underlay_map=True)
    # nusc.render_sample_data(my_sample['data']['RADAR_FRONT'], nsweeps=5, underlay_map=True)


    #In the radar plot above we only see very confident radar returns from two vehicles.
    # This is due to the filter settings defined in the file nuscenes/utils/data_classes.py.
    # If instead we want to disable all filters and render all returns,
    # we can use the disable_filters() function.
    # This returns a denser point cloud, but with many returns from background objects.
    # To return to the default settings, simply call default_filters().

    # 上面的毫米波雷达进行了过滤，可以采用下面的方法将过滤关闭
    # from nuscenes.utils.data_classes import RadarPointCloud
    # RadarPointCloud.disable_filters()
    # nusc.render_sample_data(my_sample['data']['RADAR_FRONT'], nsweeps=5, underlay_map=True)
    # RadarPointCloud.default_filters()

    #只显示某个特定ann标注内容
    # nusc.render_annotation(my_sample['anns'][22])
    nusc.render_annotation(my_sample['anns'][22], extra_info=True, out_path="./data/debug")


def debug_render_scene_map(nusc):
    # print(nusc.scene)
    my_scene_token = nusc.field2token('scene', 'name', 'scene-1100')[0]

    # The rendering command below is commented out because it may crash in notebooks
    # nusc.render_scene_channel(my_scene_token, 'CAM_FRONT')

    # The rendering command below is commented out because it may crash in notebooks
    # nusc.render_scene(my_scene_token)

    # visualize all scenes on the map for a particular location.
    nusc.render_egoposes_on_map(log_location='singapore-onenorth')


if __name__ == "__main__":
    print("Start")
    data_root = "/home/dell/liyongjing/programs/SurroundOcc-liyj/data/nuScenes_mini"
    version = 'v1.0-mini'
    # nuScenes is a large scale database that features annotated samples across 1000 scenes of approximately 20 seconds each.
    # In scenes, we annotate our data every half a second (2 Hz).
    # We define sample as an annotated keyframe of a scene at a given timestamp. A keyframe is a frame where the time-stamps of data from all the sensors should be very close to the time-stamp of the sample it points to.
    # v1.0-mini版本总共10个scene, 每个sampele大概是20s, 总共404个sample, 大概是1s两个sample
    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    # 查看scene和sample
    # debug_get_scene_sample(nusc)

    # 可视化sample的data或者anno数据
    # debug_show_sample_data(nusc)

    # 可视化场景中某个instance
    debug_show_instance_data(nusc)

    # 显示类别信息和属性信息
    # debug_show_category_attribute(nusc)

    # 显示可见属性信息
    # debug_show_visibility_data(nusc)

    # 查看sensor、calibrated_sensor、ego_pose的信息
    # debug_sensor_calibrated_sensor_ego_pose(nusc)

    # 查看log和map信息
    # debug_log_map(nusc)

    # nuscene的基本用法
    # debug_nuscene_basic(nusc)

    # 进行可视化的用法
    # debug_show_render(nusc)

    # 可视化整个场景
    # debug_render_scene_map(nusc)

    print("End")