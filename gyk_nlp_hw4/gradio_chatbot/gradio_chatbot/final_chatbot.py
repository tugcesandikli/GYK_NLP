import gradio as gr
import random
import datetime

class FinalChatbot:
    def __init__(self):
        self.greetings = [
            "Merhaba! Size nasıl yardımcı olabilirim?",
            "Selam! Bugün nasılsınız?",
            "Hoş geldiniz! Sohbet etmek için buradayım.",
            "Merhaba! Ne konuşmak istersiniz?"
        ]
        
        self.farewells = [
            "Görüşmek üzere! İyi günler!",
            "Hoşça kalın! Tekrar beklerim.",
            "Güle güle! Başka bir zaman görüşürüz.",
            "İyi günler! Yardımcı olabildiysem ne mutlu bana."
        ]
        
        self.weather_responses = [
            "Bugün hava gerçekten güzel!",
            "Hava durumu hakkında konuşmak güzel.",
            "Dışarı çıkmak için harika bir gün!",
            "Hava nasıl olursa olsun, sohbet etmek güzel."
        ]
        
        self.joke_responses = [
            "Programcılar neden karanlıkta çalışır? Çünkü ışık böcekleri çeker! 😄",
            "Bir bilgisayar bilimcisi markete gider ve 'süt' der. Kasiyer 'süt yok' der. Bilimci 'sıfır' der! 😂",
            "Neden matematikçiler Noel'i sevmez? Çünkü Noel Baba'nın vektörleri var! 🎅",
            "Bir yapay zeka bir bardağa bakar ve der: 'Bu bardak yarı dolu mu yarı boş mu?' Bardak der: 'Ben sadece bir bardakım!' 🤖"
        ]

    def get_response(self, user_input):
        """Kullanıcı mesajına uygun yanıt üretir"""
        user_input = user_input.lower().strip()
        
        # Boş mesaj kontrolü
        if not user_input:
            return "Lütfen bir mesaj yazın."
        
        # Selamlaşma
        if any(word in user_input for word in ["merhaba", "selam", "hey", "hi", "hello"]):
            return random.choice(self.greetings)
        
        # Nasılsın sorusu
        if any(word in user_input for word in ["nasılsın", "nasıl gidiyor", "iyi misin"]):
            return "Ben bir yapay zekayım, her zaman iyiyim! Size nasıl yardımcı olabilirim?"
        
        # Hava durumu
        if any(word in user_input for word in ["hava", "hava durumu", "yağmur", "güneş"]):
            return random.choice(self.weather_responses)
        
        # Şaka isteği
        if any(word in user_input for word in ["şaka", "espri", "güldür", "komik"]):
            return random.choice(self.joke_responses)
        
        # Saat sorusu
        if any(word in user_input for word in ["saat", "saat kaç", "zaman"]):
            current_time = datetime.datetime.now().strftime("%H:%M")
            return f"Şu an saat {current_time}."
        
        # Teşekkür
        if any(word in user_input for word in ["teşekkür", "teşekkürler", "sağol", "thanks"]):
            return "Rica ederim! Yardımcı olabildiysem ne mutlu bana."
        
        # Veda
        if any(word in user_input for word in ["görüşürüz", "hoşça kal", "bye", "güle güle"]):
            return random.choice(self.farewells)
        
        # Matematik işlemleri
        if any(op in user_input for op in ["+", "-", "*", "/", "topla", "çıkar", "çarp", "böl"]):
            try:
                # Basit matematik işlemleri için
                if "topla" in user_input:
                    numbers = [int(n) for n in user_input.split() if n.isdigit()]
                    if len(numbers) >= 2:
                        return f"Toplam: {sum(numbers)}"
                elif "çıkar" in user_input:
                    numbers = [int(n) for n in user_input.split() if n.isdigit()]
                    if len(numbers) >= 2:
                        return f"Fark: {numbers[0] - sum(numbers[1:])}"
                elif "çarp" in user_input:
                    numbers = [int(n) for n in user_input.split() if n.isdigit()]
                    if len(numbers) >= 2:
                        result = 1
                        for num in numbers:
                            result *= num
                        return f"Çarpım: {result}"
                elif "böl" in user_input:
                    numbers = [int(n) for n in user_input.split() if n.isdigit()]
                    if len(numbers) >= 2:
                        return f"Bölüm: {numbers[0] / numbers[1]:.2f}"
            except:
                pass
        
        # Genel yanıtlar
        general_responses = [
            f"'{user_input}' hakkında konuşmak ilginç! Daha fazla bilgi verebilir misiniz?",
            "Bu konu hakkında düşünmek güzel. Başka ne konuşmak istersiniz?",
            f"'{user_input}' dediniz. Bu konuda size nasıl yardımcı olabilirim?",
            "İlginç bir konu! Daha detaylı açıklayabilir misiniz?",
            "Bu hakkında daha fazla şey öğrenmek isterim. Devam edin!"
        ]
        
        return random.choice(general_responses)

# Chatbot örneği oluştur
chatbot = FinalChatbot()

# Basit chat fonksiyonu
def chat_with_bot(message, history):
    """Basit chat fonksiyonu - Gradio 5.x uyumlu"""
    if message.strip() == "":
        return "", history
    
    response = chatbot.get_response(message)
    return response

# Gradio arayüzü
def create_interface():
    with gr.Blocks(
        title="Final Chatbot",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 800px;
            margin: auto;
        }
        """
    ) as demo:
        gr.Markdown(
            """
            # 🤖 Final Chatbot
            
            Merhaba! Ben size yardımcı olmak için buradayım. 
            
            **Yapabileceklerim:**
            - Selamlaşma ve sohbet
            - Hava durumu hakkında konuşma
            - Şaka anlatma
            - Saat söyleme
            - Basit matematik işlemleri
            - Ve daha fazlası!
            
            Bir mesaj yazın ve sohbete başlayalım! 😊
            """
        )
        
        # Basit chat arayüzü
        chatbot_interface = gr.ChatInterface(
            fn=chat_with_bot,
            title="Sohbet Alanı",
            description="Mesajınızı yazın ve Enter'a basın",
            examples=[
                ["Merhaba!"],
                ["Nasılsın?"],
                ["Hava nasıl?"],
                ["Bana bir şaka anlat"],
                ["Saat kaç?"],
                ["5 ile 3'ü topla"]
            ]
        )
        
        gr.Markdown(
            """
            ---
            **Not:** Bu chatbot Gradio 5.x ile tam uyumlu olarak hazırlanmıştır.
            """
        )
    
    return demo

# Uygulamayı başlat
if __name__ == "__main__":
    print("🤖 Final Chatbot başlatılıyor...")
    demo = create_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
