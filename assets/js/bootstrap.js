window.$ = window.jQuery = require('jquery');

require('bootstrap-sass');

window.axios = require('axios');

window.axios.defaults.headers.common['X-Requested-With'] = 'XMLHttpRequest';