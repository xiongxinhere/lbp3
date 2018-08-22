class Classifier(object):
    """docstring for FeatureExtrctor"""
    def __init__(self, gpu_num = 1, gpus = "0"):
        self.gpu_num = gpu_num
        self.gpus = gpus

        self.model = None

    def build_model(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpus
        with tf.device('/cpu:0'):

        # build base model body
            input_tensor = Input((224, 224, 3))
            x = Lambda(vgg19.preprocess_input)(input_tensor)
        
            base_model = VGG19(input_tensor=x, weights=None, include_top=False)
        
        # build temp model for weights assign
            m_out = base_model.output
            flatten = Flatten(name='flatten')(m_out)
            fc1 = Dense(4096, activation='relu', name='fc1')(flatten)
            drop_fc1 = Dropout(1.0, name='drop_fc1')(fc1)
            fc2 = Dense(4096, activation='relu', name='fc2')(drop_fc1)
            drop_fc2 = Dropout(1.0, name='drop_fc2')(fc2)
            predictions = Dense(18, activation='softmax', name='predictions')(drop_fc2)
            
            model_weights = Model(inputs=base_model.input, outputs=predictions)
            model_weights.load_weights("vgg19_finetune.h5")
        
        # build real model for feature extraction
            m_out = base_model.output
            head_pool = AveragePooling2D(pool_size=4, strides=2, padding='valid')(m_out)
            flatten = Flatten(name='flatten')(head_pool)
            model = Model(inputs=base_model.input, outputs=flatten)
        
            for i in range(23):
                pretrained_weights = model_weights.get_layer(index=i).get_weights()
                model.get_layer(index=i).set_weights(pretrained_weights)
                
            #model.summary()
        
        parallel_model = multi_gpu_model(model, gpus = self.gpu_num)
        #parallel_model.summary()

        print("    build completed!")

        self.model = parallel_model

    def read_big_pic(self, pic_path):
        file_list = os.listdir(pic_path)
        file_list.sort()

        pics = []
        for i in range(128 * 128):
            single_pic = cv2.imread(pic_path + "/" + file_list[i])
            pics.append(single_pic)
        pics = np.array(pics)

        self.pics = pics

    def feature_extrctor(self):
        result_all = self.model.predict(self.pics, batch_size=128, verbose=1)
        result_all = np.array(result_all)
        result_all_format = result_all.reshape(128,128,2048)

        return result_all_format