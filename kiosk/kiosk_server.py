from flask import Flask, render_template, request, jsonify
import requests  # 서버와 통신하기 위한 라이브러리

app = Flask(__name__)

# 상품 리스트 저장
cart = []
total_weight = 0

# 서버 주소
SERVER_URL = 'http://172.20.10.4:5000/update_stock'

# 키오스크 서버가 상품 정보를 받는 엔드포인트
@app.route('/receive_product', methods=['POST'])
def receive_product():
    data = request.get_json()  # 서버로부터 받은 상품 정보
    product_name = data.get('product_name')
    price = data.get('product_price')
    image_url = data.get('product_image')
    weight = data.get('weight')
    # 장바구니에 상품 추가
    cart.append({"name": product_name, "price": price, "image_url": image_url, "quantity": 1 , "weight" : weight})

    print(f"상품 이름: {product_name}, 가격: {price}, 이미지: {image_url} , 무게 : {weight}")
    return jsonify({"status": "received"}), 200

@app.route('/weight', methods=['POST'])
def receive_weight():
    data = request.get_json()
    global total_weight 
    total_weight = data.get('weight')
    print(f"총 무게: {total_weight}")
    return jsonify({"status": "received"}), 200

# 현재 장바구니 상품을 가져오는 엔드포인트
@app.route('/get_cart', methods=['GET'])
def get_cart():
    return jsonify(cart)

@app.route('/clear_cart', methods=['POST'])
def clear_cart():
    global total_weight
    cart_weight = sum(item['weight'] for item in cart)
    weight_diff = abs(total_weight - cart_weight) / cart_weight

    if weight_diff <= 0.03:  # 오차가 ±3% 이내인 경우에만 결제 성공
        try:
            response = requests.post(SERVER_URL, json={"cart": cart})
            if response.status_code == 200:
                cart.clear()
                total_weight = 0  # 결제 후 무게 초기화
                return jsonify({"status": "cart cleared"}), 200
            else:
                return jsonify({"error": "재고 업데이트 실패"}), 400
        except Exception as e:
            return jsonify({"error": f"서버와 통신 중 오류 발생: {e}"}), 500
    else:
        cart.clear()  # 결제 실패 시 장바구니 초기화
        return jsonify({"status": "weight mismatch"}), 400

# 키오스크 UI 화면
@app.route('/')
def index():
    return render_template('index.html', products=cart)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # 키오스크 서버는 포트 5001에서 실행
