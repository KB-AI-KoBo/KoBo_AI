const path = require('path');
const { override } = require('customize-cra');
const webpack = require('webpack');

module.exports = override(
    (config, env) => {
        config.plugins = config.plugins.map(plugin => {
            if (plugin.constructor.name === 'HtmlWebpackPlugin') {
                // 상대 경로를 사용하여 index.html 파일 경로를 설정합니다.
                plugin.options.template = path.resolve(__dirname, 'app/Frontend/public/index.html');
            }
            return plugin;
        });
        return config;
    }
);

module.exports = function override(config) {
    config.resolve.fallback = {
        ...config.resolve.fallback,
        "path": require.resolve("path-browserify"),
        "os": require.resolve("os-browserify/browser"),
        "crypto": require.resolve("crypto-browserify"),
        "buffer": require.resolve("buffer/"),
        "stream": require.resolve("stream-browserify"),
        "process": require.resolve("process/browser"),
    };
    config.plugins = (config.plugins || []).concat([
        new webpack.ProvidePlugin({
            Buffer: ['buffer', 'Buffer'],
            process: 'process/browser',
        }),
    ]);
    return config;
};

