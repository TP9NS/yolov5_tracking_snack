from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# 상품 리스트 저장
cart = []

# 키오스크 서버가 상품 정보를 받는 엔드포인트
@app.route('/receive_product', methods=['POST'])
def receive_product():
    data = request.get_json()  # 서버로부터 받은 상품 정보
    product_name = data.get('product_name')
    price = data.get('product_price')
    image_url = data.get('product_image')

    # 장바구니에 상품 추가
    cart.append({"name": product_name, "price": price, "image_url": image_url})

    print(f"상품 이름: {product_name}, 가격: {price}, 이미지: {image_url}")
    return jsonify({"status": "received"}), 200

# 키오스크 UI 화면
@app.route('/')
def index():
    return render_template('index.html', products=cart)

@app.route('/clear_cart', methods=['POST'])
def clear_cart():
    cart.clear()  # 결제 완료 시 장바구니를 비움
    return jsonify({"status": "cart cleared"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # 키오스크 서버는 포트 5001에서 실행
