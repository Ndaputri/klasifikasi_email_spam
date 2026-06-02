 <!-- Footer -->
    <footer class="bg-white border-t border-brand-100">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
            <div class="grid md:grid-cols-4 gap-8">
                <div class="md:col-span-2">
                    <div class="flex items-center gap-2 mb-4">
                        <div class="">
                            <i class="fas fa-envelope-shield text-white text-xs"></i>
                        </div>
                        <span class="font-display font-bold text-xl text-gray-800">Klasifikasi <span class="text-gradient-pink">Email Spam</span></span>
                    </div>
                    <p class="text-gray-500 text-sm max-w-md">
                        Klasifikasi email spam bahasa Indonesia menggunakan algoritma Support Vector Machine untuk perlindungan inbox yang optimal.
                    </p>
                </div>
                <div>
                    <h4 class="font-display font-semibold text-gray-900 mb-4">Tautan Cepat</h4>
                    <ul class="space-y-2 text-sm">
                        <li><a href="#hero" class="text-gray-500 hover:text-brand-600">Beranda</a></li>
                        <li><a href="#about" class="text-gray-500 hover:text-brand-600">Tentang</a></li>
                        <li><a href="#teknologi" class="text-gray-500 hover:text-brand-600">Teknologi</a></li>
                    </ul>
                </div>
                <div>
                    <h4 class="font-display font-semibold text-gray-900 mb-4">Kontak</h4>
                    <p class="text-gray-500 text-sm flex items-center gap-2">
                        <i class="fas fa-envelope text-brand-400"></i> wandawesome@gmail.com
                    </p>
                    <p class="text-gray-500 text-sm mt-2 flex items-center gap-2">
                        <i class="fas fa-map-marker-alt text-brand-400"></i> Indonesia
                    </p>
                </div>
            </div>
            <div class="border-t border-gray-100 mt-10 pt-8 text-center">
                <p class="text-gray-400 text-sm">&copy; 2026 Wanda Putri P.W — Klasifikasi Email Spam dengan SVM</p>
            </div>
        </div>
    </footer>

    <script>
        const mobileBtn = document.getElementById('mobile-menu-btn');
        const mobileMenu = document.getElementById('mobile-menu');
        
        if(mobileBtn && mobileMenu) {
            mobileBtn.addEventListener('click', () => mobileMenu.classList.toggle('hidden'));
        }
        
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if(target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    if(mobileMenu) mobileMenu.classList.add('hidden');
                }
            });
        });
        
        window.addEventListener('scroll', () => {
            const nav = document.getElementById('navbar');
            if(window.scrollY > 20) nav.classList.add('shadow-sm');
            else nav.classList.remove('shadow-sm');
        });
    </script>

