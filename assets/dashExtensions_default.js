window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature) {
            return {
                fillColor: feature.properties.color,
                color: 'black',
                weight: 0.8,
                fillOpacity: feature.properties.block_id === -1 ? 0.5 : 0.75
            };
        }
    }
});