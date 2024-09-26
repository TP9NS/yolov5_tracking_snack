from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# 상품 리스트 저장
cart = []

@app.route('/')
def index():
    # 현재 장바구니에 있는 상품 정보를 UI에 전달
    return render_template('index.html', products=cart)

@app.route('/receive_product', methods=['POST'])
def receive_product():
    data = request.get_json()  # 상품 정보를 받음
    product_name = data.get('name')
    price = data.get('price')

    # 장바구니에 상품 추가
    cart.append({"name": product_name, "price": price})

    print(f"상품 이름: {product_name}, 가격: {price}")
    return jsonify({"status": "received"}), 200

@app.route('/clear_cart', methods=['POST'])
def clear_cart():
    # 장바구니를 비우는 함수 (결제 완료 후 사용)
    cart.clear()
    return jsonify({"status": "cart cleared"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
