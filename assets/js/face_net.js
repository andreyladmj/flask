import client from 'socket.io-client';
import Vue from 'vue';

const socket = client.connect('http://' + document.domain + ':' + location.port);

socket.on('connect', function() {
    console.info('Socket connected');
});

socket.on('disconnect', function() {
    console.info('Socket disconnected');
});

Vue.component('callout', {
    template: '#callout-template',
    delimiters: ['{(', ')}'],
    props: ['callout'],
    data() {
        //console.log('this.props.collout.type', this.props.collout.type);
        return {
            success: false,
            loading: false,
            showLogs: false,
            info: '',
            logs: ''
        }
    },
    created: function() {
        const type = this._props.callout.type;
        socket.on(`log-${type}`, this.log);
        socket.on(`finish-${type}`, this.finish);
        socket.emit('check', {name: type});
    },
    methods: {
        update: function() {
            const type = this.callout.type;
            socket.emit('update', {name: type});
            this.showLogs = true;
            this.loading = true;
            this.logs = '';
            this.info = '';
        },
        log: function(data) {
            if (data.message)
                this.logs += data.message + "<br />";

            if (data.image) {
                let arrayBufferView = new Uint8Array( data.image );
                let blob = new Blob( [ arrayBufferView ], { type: "image/jpeg" } );
                let urlCreator = window.URL || window.webkitURL;
                let imageUrl = urlCreator.createObjectURL( blob );
                this.logs += `<img src="${imageUrl}" width="50" /><br />`;
            }
        },
        finish: function(data) {
            console.log('Finish', this.callout.type, data);
            this.loading = false;

            if(data.error) {
                this.success = false;

                this.info += `<p><b style="color:#860000;">Error:</b> ${data.error}</p>`
            } else {
                this.success = true;
                //this.showLogs = false;

                for (let key in data) {
                    this.info += `<p><b>${key.capitalize()}:</b> ${data[key]}</p>`
                }
            }
        }
    }
});

new Vue({
    el: '#callout-list',
    data: {
        callouts: [
            {title:"LFW", type:"lfw"},
            {title:"Weights", type:"landmarks"},
            {title:"Model", type:"model"},
            {title:"Preprocessed", type:"output"},
            {title:"Prediction", type:"prediction"},
        ]
    }
});

String.prototype.capitalize = function() {
    return this.charAt(0).toUpperCase() + this.slice(1);
};
