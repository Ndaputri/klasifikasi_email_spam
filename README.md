
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Klasifikasi Email Spam Bahasa Indonesia - SVM</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700;14..32,800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: {
                        'sans': ['Inter', 'system-ui', 'sans-serif'],
                        'display': ['Plus Jakarta Sans', 'system-ui', 'sans-serif'],
                    },
                    colors: {
                        brand: {
                            50: '#fdf2f8',
                            100: '#fce7f3',
                            200: '#fbcfe8',
                            300: '#f9a8d4',
                            400: '#f472b6',
                            500: '#ec4899',
                            600: '#db2777',
                            700: '#be185d',
                            800: '#9d174d',
                            900: '#831843',
                            950: '#500724',
                        }
                    },
                    animation: {
                        'float': 'float 6s ease-in-out infinite',
                        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                    },
                    keyframes: {
                        float: {
                            '0%, 100%': { transform: 'translateY(0px)' },
                            '50%': { transform: 'translateY(-20px)' },
                        }
                    }
                }
            }
        }
    </script>
    <style>
        body {
            background: linear-gradient(135deg, #ffffff 0%, #fdf2f8 100%);
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(236, 72, 153, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.04);
        }
        
        .glass-card-solid {
            background: rgba(255, 255, 255, 0.98);
            border: 1px solid rgba(236, 72, 153, 0.15);
            box-shadow: 0 20px 35px -12px rgba(0, 0, 0, 0.05);
        }
        
        .gradient-pink {
            background: linear-gradient(135deg, #ec4899 0%, #be185d 100%);
        }
        
        .text-gradient-pink {
            background: linear-gradient(135deg, #ec4899 0%, #be185d 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #ec4899 0%, #be185d 100%);
            transition: all 0.3s ease;
            box-shadow: 0 4px 14px 0 rgba(236, 72, 153, 0.3);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px 0 rgba(236, 72, 153, 0.4);
        }
        
        .feature-icon {
            background: linear-gradient(135deg, #fdf2f8 0%, #fce7f3 100%);
            border: 1px solid rgba(236, 72, 153, 0.2);
        }
        
        .hover-lift {
            transition: all 0.3s ease;
        }
        
        .hover-lift:hover {
            transform: translateY(-4px);
        }
        
        .nav-blur {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid rgba(236, 72, 153, 0.1);
        }
        
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #fce7f3;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #ec4899;
            border-radius: 20px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #be185d;
        }
        
        .stat-card {
            transition: all 0.2s ease;
        }
        
        .stat-card:hover {
            border-color: #ec4899;
            box-shadow: 0 10px 25px -5px rgba(236, 72, 153, 0.1);
        }
    </style>
</head>
<body class="font-sans antialiased">

    <!-- Navigation -->
    <nav class="fixed top-0 left-0 right-0 z-50 transition-all duration-300 nav-blur" id="navbar">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between items-center h-16 lg:h-20">
                <div class="flex items-center gap-3">
                    <div class="justify-center">
                        <i class="fas fa-envelope-shield text-white text-sm"></i>
                    </div>
                    <span class="font-display font-bold text-xl text-gray-800">Klasifikasi <span class="text-gradient-pink">Email Spam</span></span>
                </div>
                
                <div class="hidden md:flex items-center gap-2">
                    <a href="#hero" class="px-4 py-2 text-gray-600 hover:text-brand-600 font-medium rounded-xl transition-all hover:bg-brand-50">Beranda</a>
                    <a href="#about" class="px-4 py-2 text-gray-600 hover:text-brand-600 font-medium rounded-xl transition-all hover:bg-brand-50">Tentang</a>
                    <a href="#technology" class="px-4 py-2 text-gray-600 hover:text-brand-600 font-medium rounded-xl transition-all hover:bg-brand-50">Teknologi</a>
                    <a href="/klasifikasi_email_spam/predict" class="ml-3 btn-primary text-white px-5 py-2.5 rounded-xl font-semibold text-sm flex items-center gap-2">
                        <i class="fas fa-microchip"></i>
                        Klasifikasi
                    </a>
                </div>
                
                <button class="md:hidden text-gray-700 p-2 rounded-lg hover:bg-brand-50" id="mobile-menu-btn">
                    <i class="fas fa-bars text-xl"></i>
                </button>
            </div>
        </div>
        
        <div class="md:hidden hidden bg-white/95 backdrop-blur-lg border-b border-brand-100" id="mobile-menu">
            <div class="px-4 py-4 space-y-2">
                <a href="#hero" class="block px-4 py-3 text-gray-600 hover:text-brand-600 hover:bg-brand-50 rounded-xl font-medium">Beranda</a>
                <a href="#about" class="block px-4 py-3 text-gray-600 hover:text-brand-600 hover:bg-brand-50 rounded-xl font-medium">Tentang</a>
                <a href="#technology" class="block px-4 py-3 text-gray-600 hover:text-brand-600 hover:bg-brand-50 rounded-xl font-medium">Teknologi</a>
                <a href="/klasifikasi_email_spam/predict" class="block btn-primary text-white px-4 py-3 rounded-xl font-semibold text-center mt-3">Klasifikasi</a>
            </div>
        </div>
    </nav>

   <!-- Hero Section -->
    <section id="hero" class="pt-28 lg:pt-36 pb-20 lg:pb-28 relative overflow-hidden">
        <div class="absolute top-0 right-0 w-[600px] h-[600px] bg-gradient-to-bl from-brand-100/40 to-transparent rounded-full blur-3xl -z-10"></div>
        <div class="absolute bottom-0 left-0 w-[500px] h-[500px] bg-gradient-to-tr from-brand-50/60 to-transparent rounded-full blur-3xl -z-10"></div>
        
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="grid lg:grid-cols-2 gap-12 items-center">
                <div class="text-center lg:text-left">
                    <div class="inline-flex items-center gap-2 bg-white/70 backdrop-blur-sm border border-brand-200 rounded-full px-4 py-1.5 mb-6 shadow-sm">
                        <span class="w-2 h-2 rounded-full bg-brand-500 animate-pulse"></span>
                        <span class="text-sm font-medium text-brand-700">Wanda Putri Prasetio Wulandari</span>
                    </div>
                    
                    <h1 class="font-display text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-gray-900 leading-tight mb-6">
                        KLASIFIKASI E-MAIL
                        <span class="text-gradient-pink">SPAM BAHASA INDONESIA</span>
                        <br>MENGGUNAKAN ALGORITMA
                        <span class="text-gradient-pink">SUPPORT VECTOR MACHINE</span>
                    </h1>
                    
                    <p class="text-lg text-gray-600 mb-8 leading-relaxed max-w-xl mx-auto lg:mx-0">
                        Sistem klasifikasi e-mail digunakan untuk memisahkan pesan <span class="font-semibold text-brand-600">spam</span> dan <span class="font-semibold text-brand-600">ham</span> 
                        berbahasa Indonesia menggunakan algoritma <span class="font-semibold text-brand-600">Support Vector Machine</span> 
                        untuk melindungi inbox dari ancaman serta untuk memenuhi Tugas Akhir Skripsi.
                    </p>
                    
                    <div class="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
                        <a href="/klasifikasi_email_spam/predict" class="btn-primary text-white px-8 py-3.5 rounded-xl font-semibold flex items-center justify-center gap-3 group">
                            <span>Mulai Klasifikasi</span>
                            <i class="fas fa-arrow-right group-hover:translate-x-1 transition-transform"></i>
                        </a>
                    </div>
                </div>
                
                <div class="hidden lg:block">
                        <div class="flex items-center gap-3 mb-4">
                            </div>
                         <div class="order-1 lg:order-2">
                    <div class="glass-card-solid rounded-3xl p-8">
                        <div class="flex items-center justify-between mb-6">
                            <h3 class="font-display font-bold text-xl text-gray-800">Alur Penelitian</h3>
                            <i class="fas fa-project-diagram text-brand-400 text-2xl"></i>
                        </div>
                        <div class="space-y-4">
                            <div class="flex items-center gap-3">
                                <div class="w-8 h-8 rounded-full bg-brand-100 flex items-center justify-center text-brand-600 font-bold text-sm">1</div>
                                <div class="flex-1">
                                    <div class="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                                        <div class="h-full w-full bg-brand-400 rounded-full"></div>
                                    </div>
                                    <p class="text-xs text-gray-500 mt-1">Handle missing values, cleaning, case folding, normalisasi, tokenisasi, stopword removal, stemming, dan remove duplicate.</p>
                                </div>
                            </div>
                            <div class="flex items-center gap-3">
                                <div class="w-8 h-8 rounded-full bg-brand-100 flex items-center justify-center text-brand-600 font-bold text-sm">2</div>
                                <div class="flex-1">
                                    <div class="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                                        <div class="h-full w-11/12 bg-brand-500 rounded-full"></div>
                                    </div>
                                    <p class="text-xs text-gray-500 mt-1">TF-IDF Vectorization Mengubah teks menjadi representasi numerik dengan mempertimbangkan bobot kata.</p>
                                </div>
                            </div>
                            <div class="flex items-center gap-3">
                                <div class="w-8 h-8 rounded-full bg-brand-100 flex items-center justify-center text-brand-600 font-bold text-sm">3</div>
                                <div class="flex-1">
                                    <div class="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                                        <div class="h-full w-full bg-brand-600 rounded-full"></div>
                                    </div>
                                    <p class="text-xs text-gray-500 mt-1">LinearSVC Classification Proses pelatihan dan prediksi menggunakan algoritma Support Vector Machine dengan kernel linear.</p>
                                </div>
                            </div>
                            <div class="flex items-center gap-3">
                                <div class="w-8 h-8 rounded-full bg-brand-100 flex items-center justify-center text-brand-600 font-bold text-sm">4</div>
                                <div class="flex-1">
                                    <div class="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                                        <div class="h-full w-10/12 bg-brand-700 rounded-full"></div>
                                    </div>
                                    <p class="text-xs text-gray-500 mt-1">Evaluasi model : Pengukuran performa menggunakan akurasi, presisi, recall, dan F1-Score.</p>
                                </div>
                            </div>
                
                        </div>
                    </div>
                </div>
           
        </div>
  

    <!-- About Section -->
    <section id="about" class="py-20 bg-white/50 relative">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="text-center max-w-3xl mx-auto mb-16">
                <div class="inline-flex items-center gap-2 bg-brand-100 rounded-full px-4 py-1.5 mb-4">
                    <i class="fas fa-info-circle text-brand-600 text-sm"></i>
                    <span class="text-sm font-medium text-brand-700">Tentang Sistem</span>
                </div>
                <h2 class="font-display text-4xl lg:text-5xl font-bold text-gray-900 mb-5">
                    Klasifikasi E-mail Spam Bahasa Indonesia
                </h2>
                <p class="text-gray-600 text-lg leading-relaxed">
                    Sistem mendeteksi dan mengklasifikasi email spam bahasa Indonesia yang dirancang untuk melindungi dari ancaman email berbahaya.
                </p>
            </div>
            
            <div class="grid md:grid-cols-3 gap-8">
                <div class="glass-card-solid rounded-2xl p-7 hover-lift">
                    <div class="w-14 h-14 feature-icon rounded-xl flex items-center justify-center mb-5">
                        <i class="fas fa-language text-brand-500 text-2xl"></i>
                    </div>
                    <h3 class="font-display font-bold text-xl text-gray-800 mb-3">Bahasa Indonesia</h3>
                    <p class="text-gray-500 leading-relaxed">Mendukung teks berbahasa Indonesia dengan stemming Sastrawi dan stopword removal untuk hasil optimal.</p>
                </div>
                <div class="glass-card-solid rounded-2xl p-7 hover-lift">
                    <div class="w-14 h-14 feature-icon rounded-xl flex items-center justify-center mb-5">
                        <i class="fas fa-chart-line text-brand-500 text-2xl"></i>
                    </div>
                    <h3 class="font-display font-bold text-xl text-gray-800 mb-3">Akurasi Tinggi</h3>
                    <p class="text-gray-500 leading-relaxed">Model LinearSVC yang dioptimalkan dengan TF-IDF vectorizer menghasilkan Akurasi, Presisi, Recall & F1 Score yang optimal.</p>
                </div>
                <div class="glass-card-solid rounded-2xl p-7 hover-lift">
                    <div class="w-14 h-14 feature-icon rounded-xl flex items-center justify-center mb-5">
                        <i class="fas fa-shield-virus text-brand-500 text-2xl"></i>
                    </div>
                    <h3 class="font-display font-bold text-xl text-gray-800 mb-3">Privasi Terjaga</h3>
                    <p class="text-gray-500 leading-relaxed">Data email tidak disimpan permanen, hanya diproses untuk keperluan klasifikasi.</p>
                </div>
            </div>
        </div>
    </section>

    <!-- Technology Section -->
    <section id="technology" class="py-20 relative">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="text-center mb-16">
                <div class="inline-flex items-center gap-2 bg-brand-100 rounded-full px-4 py-1.5 mb-4">
                    <i class="fas fa-microchip text-brand-600"></i>
                    <span class="text-sm font-medium text-brand-700">Inti Teknologi</span>
                </div>
                <h2 class="font-display text-4xl lg:text-5xl font-bold text-gray-900 mb-5">
                    Algoritma <span class="text-gradient-pink">Support Vector Machine</span>
                </h2>
                <p class="text-gray-600 text-lg max-w-2xl mx-auto">
                    Menggunakan LinearSVC yang unggul dalam klasifikasi teks dimensi tinggi dengan margin pemisah maksimal.
                </p>
            </div>
            
            <div class="grid lg:grid-cols-2 gap-12 items-center">
                <div>
                    <div class="space-y-6">
                        <div class="flex gap-4 p-5 glass-card-solid rounded-2xl">
                            <div class="flex-shrink-0 w-12 h-12 bg-brand-100 rounded-xl flex items-center justify-center">
                                <i class="fas fa-chalkboard-teacher text-brand-600 text-xl"></i>
                            </div>
                            <div>
                                <h4 class="font-display font-bold text-gray-800 mb-1">Pembelajaran Supervised</h4>
                                <p class="text-gray-500 text-sm">Model dilatih dengan dataset email berlabel spam & ham untuk memahami pola teks mencurigakan.</p>
                            </div>
                        </div>
                        <div class="flex gap-4 p-5 glass-card-solid rounded-2xl">
                            <div class="flex-shrink-0 w-12 h-12 bg-brand-100 rounded-xl flex items-center justify-center">
                                <i class="fas fa-chart-pie text-brand-600 text-xl"></i>
                            </div>
                            <div>
                                <h4 class="font-display font-bold text-gray-800 mb-1">Ekstraksi Fitur TF-IDF</h4>
                                <p class="text-gray-500 text-sm">Mengubah teks menjadi vektor numerik dengan mempertimbangkan frekuensi kata dan kebalikan frekuensi dokumen.</p>
                            </div>
                        </div>
                        <div class="flex gap-4 p-5 glass-card-solid rounded-2xl">
                            <div class="flex-shrink-0 w-12 h-12 bg-brand-100 rounded-xl flex items-center justify-center">
                                <i class="fas fa-chart-line text-brand-600 text-xl"></i>
                            </div>
                            <div>
                                <h4 class="font-display font-bold text-gray-800 mb-1">Evaluasi Model</h4>
                                <p class="text-gray-500 text-sm">Pengujian menggunakan confusion matrix, akurasi, presisi, recall, dan F1-Score.</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="glass-card-solid rounded-3xl p-8">
                    <div class="flex items-center justify-between mb-6">
                        <h3 class="font-display font-bold text-xl text-gray-800">Teknologi Pendukung</h3>
                        <i class="fas fa-cogs text-brand-400 text-2xl"></i>
                    </div>
                    <div class="grid grid-cols-2 gap-4">
                        <div class="text-center p-3 bg-brand-50 rounded-xl">
                            <i class="fab fa-python text-2xl text-brand-500"></i>
                            <p class="text-xs text-gray-600 mt-1">Python</p>
                        </div>
                        <div class="text-center p-3 bg-brand-50 rounded-xl">
                            <i class="fas fa-brain text-2xl text-brand-500"></i>
                            <p class="text-xs text-gray-600 mt-1">Scikit-learn</p>
                        </div>
                        <div class="text-center p-3 bg-brand-50 rounded-xl">
                            <i class="fas fa-flask text-2xl text-brand-500"></i>
                            <p class="text-xs text-gray-600 mt-1">Flask</p>
                        </div>
                        <div class="text-center p-3 bg-brand-50 rounded-xl">
                            <i class="fas fa-file-alt text-2xl text-brand-500"></i>
                            <p class="text-xs text-gray-600 mt-1">TF-IDF</p>
                        </div>
                        <div class="text-center p-3 bg-brand-50 rounded-xl">
                            <i class="fas fa-language text-2xl text-brand-500"></i>
                            <p class="text-xs text-gray-600 mt-1">Sastrawi</p>
                        </div>
                        <div class="text-center p-3 bg-brand-50 rounded-xl">
                            <i class="fas fa-chart-bar text-2xl text-brand-500"></i>
                            <p class="text-xs text-gray-600 mt-1">Pandas</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>


   <!-- Footer -->
<footer class="bg-white border-t border-brand-100">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div class="grid md:grid-cols-4 gap-8">
            <!-- Kolom 1: Brand -->
            <div class="md:col-span-2">
                <div class="flex items-center gap-2 mb-4">
                    <div class="w-9 h-9 gradient-pink rounded-xl flex items-center justify-center shadow-md">
                        <i class="fas fa-envelope-shield text-white text-sm"></i>
                    </div>
                    <span class="font-display font-bold text-xl text-gray-800">Klasifikasi <span class="text-gradient-pink">Email Spam</span></span>
                </div>
                <p class="text-gray-500 text-sm max-w-md leading-relaxed">
                    Klasifikasi email spam bahasa Indonesia menggunakan algoritma <span class="font-semibold text-brand-600">Support Vector Machine</span> 
                    untuk perlindungan inbox yang optimal dan memenuhi Tugas Akhir Skripsi.
                </p>
                <div class="flex gap-3 mt-5">
                    <a href="#" class="w-8 h-8 bg-brand-50 rounded-full flex items-center justify-center text-brand-500 hover:bg-brand-500 hover:text-white transition-all duration-300">
                        <i class="fab fa-github text-sm"></i>
                    </a>
                    <a href="#" class="w-8 h-8 bg-brand-50 rounded-full flex items-center justify-center text-brand-500 hover:bg-brand-500 hover:text-white transition-all duration-300">
                        <i class="fab fa-linkedin-in text-sm"></i>
                    </a>
                    <a href="#" class="w-8 h-8 bg-brand-50 rounded-full flex items-center justify-center text-brand-500 hover:bg-brand-500 hover:text-white transition-all duration-300">
                        <i class="fab fa-instagram text-sm"></i>
                    </a>
                </div>
            </div>
            
            <!-- Kolom 2: Tautan Cepat -->
            <div>
                <h4 class="font-display font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <i class="fas fa-link text-brand-400 text-sm"></i>
                    Tautan Cepat
                </h4>
                <ul class="space-y-2 text-sm">
                    <li><a href="#hero" class="text-gray-500 hover:text-brand-600 transition-colors flex items-center gap-2"><i class="fas fa-chevron-right text-brand-300 text-xs"></i>Beranda</a></li>
                    <li><a href="#about" class="text-gray-500 hover:text-brand-600 transition-colors flex items-center gap-2"><i class="fas fa-chevron-right text-brand-300 text-xs"></i>Tentang</a></li>
                    <li><a href="#technology" class="text-gray-500 hover:text-brand-600 transition-colors flex items-center gap-2"><i class="fas fa-chevron-right text-brand-300 text-xs"></i>Teknologi</a></li>
                    <li><a href="#" class="text-gray-500 hover:text-brand-600 transition-colors flex items-center gap-2"><i class="fas fa-chevron-right text-brand-300 text-xs"></i>Klasifikasi</a></li>
                </ul>
            </div>
            
            <!-- Kolom 3: Kontak & Info -->
            <div>
                <h4 class="font-display font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <i class="fas fa-address-card text-brand-400 text-sm"></i>
                    Kontak
                </h4>
                <ul class="space-y-3 text-sm">
                    <li class="flex items-center gap-3 text-gray-500">
                        <i class="fas fa-envelope text-brand-400 w-4"></i>
                        <span>wandawesome@gmail.com</span>
                    </li>
                    <li class="flex items-center gap-3 text-gray-500">
                        <i class="fas fa-map-marker-alt text-brand-400 w-4"></i>
                        <span>Indonesia</span>
                    </li>
                    <li class="flex items-center gap-3 text-gray-500">
                        <i class="fas fa-graduation-cap text-brand-400 w-4"></i>
                        <span>Tugas Akhir Skripsi</span>
                    </li>
                </ul>
            </div>
        </div>
        
        <!-- Bottom Bar -->
        <div class="border-t border-gray-100 mt-10 pt-8">
            <div class="flex flex-col md:flex-row justify-between items-center gap-4">
                <p class="text-gray-400 text-sm">
                    &copy; 2026 <span class="font-semibold text-brand-500">Wanda Putri P.W</span> — Klasifikasi Email Spam dengan SVM
                </p>
                <div class="flex gap-4 text-xs text-gray-400">
                    <a href="#" class="hover:text-brand-500 transition-colors">Kebijakan Privasi</a>
                    <span>|</span>
                    <a href="#" class="hover:text-brand-500 transition-colors">Syarat & Ketentuan</a>
                </div>
            </div>
        </div>
    </div>
</footer>
