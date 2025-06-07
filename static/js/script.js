document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('search-query');
    const searchBtn = document.getElementById('search-btn');
    const searchResults = document.getElementById('search-results');

    // 搜索功能
    searchBtn.addEventListener('click', performSearch);
    searchInput.addEventListener('keyup', function(event) {
        if (event.key === 'Enter') {
            performSearch();
        }
    });

    function performSearch() {
        const query = searchInput.value.trim();
        if (!query) {
            searchResults.style.display = 'none';
            return;
        }

        fetch(`/search?query=${encodeURIComponent(query)}`)
            .then(response => response.json())
            .then(data => {
                if (data.length === 0) {
                    searchResults.innerHTML = '<div class="search-result-item">没有找到匹配的电影</div>';
                    searchResults.style.display = 'block';
                    return;
                }

                searchResults.innerHTML = '';
                data.forEach(movie => {
                    const item = document.createElement('div');
                    item.className = 'search-result-item';
                    item.innerHTML = `
                        <div class="movie-title">${movie.title}</div>
                        <div class="movie-genres">${movie.genres}</div>
                    `;
                    searchResults.appendChild(item);
                });
                searchResults.style.display = 'block';
            })
            .catch(error => {
                console.error('搜索出错:', error);
                searchResults.innerHTML = '<div class="search-result-item">搜索时出错</div>';
                searchResults.style.display = 'block';
            });
    }

    // 点击搜索结果外部时隐藏结果
    document.addEventListener('click', function(event) {
        if (!searchResults.contains(event.target) && event.target !== searchInput && event.target !== searchBtn) {
            searchResults.style.display = 'none';
        }
    });
});