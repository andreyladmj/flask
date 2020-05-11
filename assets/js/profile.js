import Vue from 'vue';

let Webcam = require('./webcam.min');
let Events = new Vue({});

Vue.component('profile-images', new Vue({
    el: '#profile-images',
    props: {
        loading: false,
        text: false,
        images: []
    },
    mounted () {
        Events.$on('process-profile-image', () => {
            console.log('process-profile-image');
            this.loading = true;
        });
    },
    methods: {
        handleProfileImage: function(path) {
            this.images.push(path)
        },
        isLoading: function(is_load) {
            this.loading = !!is_load;
        }
    }
}));

Vue.component('profile-webcam', new Vue({
    el: '#webcam',
    data: {
        show: true,
        image_processing: false,
        message: ''
    },
    created: function() {
        navigator.mediaDevices.getUserMedia({
            video: true
        }).then(this.webcam_init).catch(this.webcam_not_found);
    },
    methods: {
        webcam_not_found: function() {
            this.show = false;
            console.warn('webcam not found');
        },
        webcam_init: function() {
            Webcam.set({
                width: 320,
                height: 240,
                image_format: 'jpeg',
                jpeg_quality: 90
            });
            Webcam.attach('#my_camera');
        },
        take_snapshot: function (e) {
            this.image_processing = true;
            Events.$emit('process-profile-image');
            Webcam.snap(this.upload_image);
        },
        upload_image: function (data_uri) {
            let formData = new FormData();
            formData.append("fileToUpload", data_uri);

            $.ajax({
                url: "/upload_profile_image",
                type: "POST",
                dataType: 'json',
                data: formData,
                processData: false,
                contentType: false,
                success: (response) => {
                    this.image_processing = false;

                    if (response.success) {
                        this.message = '<div class="alert alert-success">Photo Added!</div>';
                        this.$emit('profileImage', { path: response.path });
                    } else {
                        this.message = `<div class="alert alert-danger">${response.message}</div>`;
                        //_this.message = '<div class="alert alert-danger">Please, put your face straight at the webcam!</div>';
                    }
                },
                error: (jqXHR, textStatus, errorMessage) => {
                    this.image_processing = false;
                    console.error(errorMessage);
                }
            });
        }
    }
}));
