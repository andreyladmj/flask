// let Webcam = require('./webcam.min');
// let Vue = require('vue');
// new Vue({
//     el: '#login',
//     data: {
//         points: 500,
//         count: 0,
//         message: ''
//     },
//     methods: {
//         handleIt: function (e) {
//             //e.preventDefault();
//         },
//         doSomething: function () {
//             console.log('clicked');
//             this.count++;
//         }
//     },
//     computed: {
//         skill: function() {
//             if(this.points < 100) {
//                 return 'Beginner';
//             }
//
//             return 'Advanced';
//         }
//     },
//     watch: {
//         points: function(points) {
//             console.log(`Points changed up to ${points}`);
//         }
//     },
//     components: {
//         // mycounter: {
//         //     //template: `<h1>Hi </h1>`
//         //     template: '#counter-template',
//         //     props: ['subject'],
//         //     data: function() {
//         //         return { count: 0 }
//         //     }
//         // }
//     }
// });
//<!-- Configure a few settings and attach camera -->
if ($('.login').length) {
    Webcam.set({
        width: 320,
        height: 240,
        image_format: 'jpeg',
        jpeg_quality: 90
    });
    Webcam.attach( '#my_camera' );
    $('#take_snapshot').click(function(){
        console.log('take_snapshot');
        Webcam.snap( function(data_uri) {
            $('#face_photo').val(data_uri).closest('form').submit();
            console.log('submit');

        });
    });

}